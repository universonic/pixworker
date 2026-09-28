use crate::utils::ntsc::NTSC;
use anyhow::{Context, Result, bail, ensure};
use ffmpeg::{Dictionary, Packet, Rational, codec, filter, format, frame, media};
use ffmpeg_next as ffmpeg;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::{Builder, NamedTempFile};

#[derive(Clone, Copy, Debug)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub rate: NTSC,
    pub has_audio: bool,
    pub frame_estimate: Option<u64>,
}

impl VideoInfo {
    pub fn open(input: &Path) -> Result<Self> {
        ffmpeg::init()?;
        let context = format::input(input)?;
        Self::from_context(&context)
    }

    fn from_context(context: &format::context::Input) -> Result<Self> {
        let stream = context
            .streams()
            .find(|s| s.parameters().medium() == media::Type::Video)
            .context("Failed to retrieve video information from input file.")?;
        let parameters = stream.parameters();
        let rate = stream.rate();
        let (num, den) = (rate.numerator(), rate.denominator());
        ensure!(
            num > 0 && den > 0,
            "Failed to retrieve video information from input file."
        );
        let decoder = codec::context::Context::from_parameters(parameters)?
            .decoder()
            .video()?;
        ensure!(
            decoder.width() > 0 && decoder.height() > 0,
            "Failed to retrieve video information from input file."
        );
        Ok(Self {
            width: decoder.width(),
            height: decoder.height(),
            rate: NTSC {
                num: num as u64,
                den: den as u64,
            },
            has_audio: context
                .streams()
                .any(|s| s.parameters().medium() == media::Type::Audio),
            frame_estimate: (stream.frames() > 0).then_some(stream.frames() as u64),
        })
    }
}

pub struct VideoDecoder {
    input: format::context::Input,
    stream: usize,
    decoder: codec::decoder::Video,
    graph: filter::Graph,
    rate: NTSC,
    next_pts: i64,
    last: Option<frame::Video>,
    previous: Option<frame::Video>,
    repeat_previous: i64,
    repeat_current: i64,
    frames_prev_hist: [i64; 3],
    demux_eof: bool,
    filter_eof: bool,
    done: bool,
}

impl VideoDecoder {
    pub fn new(input: &Path) -> Result<Self> {
        ffmpeg::init()?;
        let input = format::input(input)?;
        let stream = input
            .streams()
            .find(|s| s.parameters().medium() == media::Type::Video)
            .context("Failed to retrieve video information from input file.")?;
        let index = stream.index();
        let rate = stream.rate();
        ensure!(
            rate.numerator() > 0 && rate.denominator() > 0,
            "Failed to retrieve video information from input file."
        );
        let time_base = stream.time_base();
        let mut context = codec::context::Context::from_parameters(stream.parameters())?;
        context.set_threading(codec::threading::Config::kind(
            codec::threading::Type::Frame,
        ));
        let decoder = context.decoder().video()?;
        let mut graph = filter::Graph::new();
        let args = format!(
            "video_size={}x{}:pix_fmt={}:time_base={}:pixel_aspect={}:frame_rate={}",
            decoder.width(),
            decoder.height(),
            decoder.format() as i32,
            time_base,
            decoder.aspect_ratio(),
            rate
        );
        graph.add(
            &filter::find("buffer").context("Missing buffer filter")?,
            "in",
            &args,
        )?;
        graph.add(
            &filter::find("buffersink").context("Missing buffersink filter")?,
            "out",
            "",
        )?;
        graph
            .output("in", 0)?
            .input("out", 0)?
            .parse("format=rgb24")?;
        // This controls auto-inserted format conversion filters, including YUV -> RGB.
        unsafe {
            let opts = std::ffi::CString::new("flags=bicubic")?;
            (*graph.as_mut_ptr()).scale_sws_opts = ffmpeg::ffi::av_strdup(opts.as_ptr());
        }
        graph.validate()?;
        Ok(Self {
            input,
            stream: index,
            decoder,
            graph,
            rate: NTSC {
                num: rate.numerator() as u64,
                den: rate.denominator() as u64,
            },
            next_pts: 0,
            last: None,
            previous: None,
            repeat_previous: 0,
            repeat_current: 0,
            frames_prev_hist: [0; 3],
            demux_eof: false,
            filter_eof: false,
            done: false,
        })
    }

    fn sync_frame(&mut self, rgb: frame::Video) {
        let tb = self.graph.get("out").unwrap().sink().time_base();
        let tick = self.rate.num as f64 / self.rate.den as f64 * tb.numerator() as f64
            / tb.denominator() as f64;
        let duration = if rgb.packet().duration == 0 {
            ffmpeg::Rescale::rescale(&1, Rational(self.rate.den as i32, self.rate.num as i32), tb)
                as f64
                * tick
        } else {
            rgb.packet().duration as f64 * tick
        };
        // ffmpeg_filter.c:adjust_frame_pts_to_encoder_tb preserves fractional ticks
        // before video_sync_process decides which frame occupies each CFR slot.
        let exact = rgb.pts().map_or(self.next_pts as f64, |pts| {
            let bits = (29 - (self.rate.num as u32).ilog2() as i32).clamp(0, 16) as u32;
            let tb_out = Rational(self.rate.den as i32, (self.rate.num << bits) as i32);
            let precise = ffmpeg::Rescale::rescale(&pts, tb, tb_out) as f64 / (1u64 << bits) as f64;
            if precise != precise.round_ties_even() {
                precise + precise.signum() / 131072.0
            } else {
                precise
            }
        });
        let mut delta0 = exact - self.next_pts as f64;
        let delta = delta0 + duration;
        if delta0 < 0.0 && delta > 0.0 {
            delta0 = 0.0;
        }
        let mut count = 1;
        let mut from_previous = 0;
        if delta < -1.1 {
            count = 0;
        } else if delta > 1.1 {
            count = (delta as f32).round_ties_even() as i64;
            if delta0 > 1.1 {
                from_previous = ((delta0 - 0.6) as f32).round_ties_even() as i64;
            }
        }
        self.frames_prev_hist.rotate_right(1);
        self.frames_prev_hist[0] = from_previous;
        self.previous = self.last.replace(rgb);
        self.repeat_previous = if self.previous.is_some() {
            from_previous.min(count)
        } else {
            0
        };
        self.repeat_current = count - self.repeat_previous;
    }

    pub fn next_frame(&mut self) -> Result<Option<frame::Video>> {
        loop {
            if self.repeat_previous > 0 || self.repeat_current > 0 {
                let source = if self.repeat_previous > 0 {
                    self.repeat_previous -= 1;
                    self.previous.as_ref().unwrap()
                } else {
                    self.repeat_current -= 1;
                    self.last.as_ref().unwrap()
                };
                let mut output = frame::Video::empty();
                let code =
                    unsafe { ffmpeg::ffi::av_frame_ref(output.as_mut_ptr(), source.as_ptr()) };
                if code < 0 {
                    return Err(ffmpeg::Error::from(code).into());
                }
                output.set_pts(Some(self.next_pts));
                self.next_pts += 1;
                return Ok(Some(output));
            }
            if self.done {
                return Ok(None);
            }
            let mut rgb = frame::Video::empty();
            match self.graph.get("out").unwrap().sink().frame(&mut rgb) {
                Ok(()) => {
                    self.sync_frame(rgb);
                    continue;
                }
                Err(ffmpeg::Error::Eof) => {
                    let mut hist = self.frames_prev_hist;
                    hist.sort_unstable();
                    self.repeat_current = if self.last.is_some() { hist[1] } else { 0 };
                    self.done = true;
                    continue;
                }
                Err(e) if again(e) => {}
                Err(e) => return Err(e.into()),
            }
            if self.filter_eof {
                bail!("Video filter did not terminate after flush");
            }
            let mut decoded = frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let pts = decoded.timestamp();
                    decoded.set_pts(pts);
                    self.graph.get("in").unwrap().source().add(&decoded)?;
                }
                Err(ffmpeg::Error::Eof) => {
                    self.graph.get("in").unwrap().source().flush()?;
                    self.filter_eof = true;
                }
                Err(e) if again(e) && !self.demux_eof => loop {
                    let mut packet = Packet::empty();
                    match packet.read(&mut self.input) {
                        Ok(()) if packet.stream() == self.stream => {
                            self.decoder.send_packet(&packet)?;
                            break;
                        }
                        Ok(()) => {}
                        Err(ffmpeg::Error::Eof) => {
                            self.decoder.send_eof()?;
                            self.demux_eof = true;
                            break;
                        }
                        Err(e) => return Err(e.into()),
                    }
                },
                Err(e) => return Err(e.into()),
            }
        }
    }
}

impl Iterator for VideoDecoder {
    type Item = Result<frame::Video>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_frame().transpose()
    }
}

struct Audio {
    input: format::context::Input,
    stream: usize,
    input_time_base: Rational,
    video_stream: Option<(usize, Rational)>,
    decoder: codec::decoder::Audio,
    graph: filter::Graph,
    encoder: codec::encoder::audio::Encoder,
    output_stream: usize,
    next_packet: Option<Packet>,
    next_video: Option<Packet>,
    input_eof: bool,
    eof: bool,
}

impl Audio {
    fn new(input: &Path, output: &mut format::context::Output) -> Result<Option<Self>> {
        let input = format::input(input)?;
        let Some(stream) = input
            .streams()
            .find(|s| s.parameters().medium() == media::Type::Audio)
        else {
            return Ok(None);
        };
        let index = stream.index();
        let input_time_base = stream.time_base();
        let decoder = codec::context::Context::from_parameters(stream.parameters())?
            .decoder()
            .audio()?;
        let layout = if decoder.channel_layout().is_empty() {
            ffmpeg::ChannelLayout::default(i32::from(decoder.channels()))
        } else {
            decoder.channel_layout()
        };
        ensure!(!layout.is_empty(), "Audio stream has no channel layout");
        let video_stream = input
            .streams()
            .find(|s| s.parameters().medium() == media::Type::Video)
            .map(|s| (s.index(), s.time_base()));
        let pcm = ffmpeg::encoder::find(codec::Id::PCM_S16LE)
            .context("pcm_s16le encoder is unavailable")?;
        let global = output
            .format()
            .flags()
            .contains(format::Flags::GLOBAL_HEADER);
        let mut out_stream = output.add_stream(pcm)?;
        let output_stream = out_stream.index();
        let mut encoder = codec::context::Context::new_with_codec(pcm)
            .encoder()
            .audio()?;
        encoder.set_rate(44100);
        encoder.set_format(ffmpeg::format::Sample::I16(
            ffmpeg::format::sample::Type::Packed,
        ));
        encoder.set_channel_layout(ffmpeg::ChannelLayout::STEREO);
        encoder.set_time_base((1, 44100));
        if global {
            encoder.set_flags(codec::Flags::GLOBAL_HEADER);
        }
        out_stream.set_time_base((1, 44100));
        let encoder = encoder.open_as(pcm)?;
        out_stream.set_parameters(&encoder);

        let mut graph = filter::Graph::new();
        let args = format!(
            "time_base={}:sample_rate={}:sample_fmt={}:channel_layout=0x{:x}",
            input_time_base,
            decoder.rate(),
            decoder.format().name(),
            layout.bits()
        );
        graph.add(
            &filter::find("abuffer").context("Missing abuffer filter")?,
            "in",
            &args,
        )?;
        graph.add(
            &filter::find("abuffersink").context("Missing abuffersink filter")?,
            "out",
            "",
        )?;
        graph.output("in", 0)?.input("out", 0)?.parse(
            "aresample=async=1:first_pts=0,aformat=sample_fmts=s16:sample_rates=44100:channel_layouts=stereo"
        )?;
        graph.validate()?;
        Ok(Some(Self {
            input,
            stream: index,
            input_time_base,
            video_stream,
            decoder,
            graph,
            encoder,
            output_stream,
            next_packet: None,
            next_video: None,
            input_eof: false,
            eof: false,
        }))
    }

    fn drain_packets(&mut self, output: &mut format::context::Output) -> Result<()> {
        let time_base = output.stream(self.output_stream).unwrap().time_base();
        loop {
            let mut packet = Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(self.output_stream);
                    packet.rescale_ts(self.encoder.time_base(), time_base);
                    packet.write_interleaved(output)?;
                }
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(e) if again(e) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn drain_filter(&mut self, output: &mut format::context::Output) -> Result<()> {
        loop {
            let mut filtered = frame::Audio::empty();
            match self.graph.get("out").unwrap().sink().frame(&mut filtered) {
                Ok(()) => {
                    // The sink's time base may differ from the encoder's time base.
                    let base = self.graph.get("out").unwrap().sink().time_base();
                    let pts = filtered
                        .pts()
                        .map(|pts| ffmpeg::Rescale::rescale(&pts, base, self.encoder.time_base()));
                    filtered.set_pts(pts);
                    self.encoder.send_frame(&filtered)?;
                    self.drain_packets(output)?;
                }
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(e) if again(e) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn drain_decoder(&mut self, output: &mut format::context::Output) -> Result<()> {
        loop {
            let mut decoded = frame::Audio::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let pts = decoded.timestamp();
                    decoded.set_pts(pts);
                    if decoded.channel_layout().is_empty() {
                        decoded.set_channel_layout(ffmpeg::ChannelLayout::default(i32::from(
                            decoded.channels(),
                        )));
                    }
                    self.graph.get("in").unwrap().source().add(&decoded)?;
                    self.drain_filter(output)?;
                }
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(e) if again(e) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn video_ahead(&self, packet: &Packet, until: Option<(i64, NTSC)>) -> bool {
        let Some((frame, fps)) = until else {
            return false;
        };
        self.video_stream.is_some_and(|(index, tb)| {
            packet.stream() == index
                && packet
                    .dts()
                    .or_else(|| packet.pts())
                    .is_some_and(|ts| later_than_frame(ts, tb, frame, fps))
        })
    }

    fn read_packet(&mut self, until: Option<(i64, NTSC)>) -> Result<Option<Packet>> {
        if let Some(packet) = self.next_video.take()
            && self.video_ahead(&packet, until)
        {
            self.next_video = Some(packet);
            return Ok(None);
        }
        if self.input_eof {
            return Ok(None);
        }
        loop {
            let mut packet = Packet::empty();
            match packet.read(&mut self.input) {
                Ok(()) if packet.stream() == self.stream => return Ok(Some(packet)),
                Ok(()) if self.video_ahead(&packet, until) => {
                    self.next_video = Some(packet);
                    return Ok(None);
                }
                Ok(()) => {}
                Err(ffmpeg::Error::Eof) => {
                    self.input_eof = true;
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn until(&mut self, output: &mut format::context::Output, frame: i64, fps: NTSC) -> Result<()> {
        while !self.eof {
            let packet = match self.next_packet.take() {
                Some(packet) => Some(packet),
                None => self.read_packet(Some((frame, fps)))?,
            };
            let Some(packet) = packet else {
                if self.input_eof {
                    self.flush(output)?;
                }
                break;
            };
            let pts = packet.pts().or_else(|| packet.dts()).unwrap_or(0);
            if later_than_frame(pts, self.input_time_base, frame, fps) {
                self.next_packet = Some(packet);
                break;
            }
            self.decoder.send_packet(&packet)?;
            self.drain_decoder(output)?;
        }
        Ok(())
    }

    fn flush(&mut self, output: &mut format::context::Output) -> Result<()> {
        if self.eof {
            return Ok(());
        }
        while let Some(packet) = match self.next_packet.take() {
            Some(packet) => Some(packet),
            None => self.read_packet(None)?,
        } {
            self.decoder.send_packet(&packet)?;
            self.drain_decoder(output)?;
        }
        self.decoder.send_eof()?;
        self.drain_decoder(output)?;
        self.graph.get("in").unwrap().source().flush()?;
        self.drain_filter(output)?;
        self.encoder.send_eof()?;
        self.drain_packets(output)?;
        self.eof = true;
        Ok(())
    }
}

pub struct VideoEncoder {
    output: PathBuf,
    mux: Option<format::context::Output>,
    temp: Option<NamedTempFile>,
    encoder: codec::encoder::video::Encoder,
    scaler: ffmpeg::software::scaling::Context,
    audio: Option<Audio>,
    fps: NTSC,
    frames: i64,
    first_video_packet: Option<Instant>,
    width: u32,
    height: u32,
}

impl VideoEncoder {
    pub fn new(
        output: &Path,
        input: &Path,
        width: u32,
        height: u32,
        fps: NTSC,
        silent: bool,
    ) -> Result<Self> {
        ffmpeg::init()?;
        ffmpeg::log::set_level(if silent {
            ffmpeg::log::Level::Error
        } else {
            ffmpeg::log::Level::Info
        });
        ensure!(
            width > 0 && height > 0 && fps.num > 0 && fps.den > 0,
            "Invalid output video dimensions or frame rate"
        );
        let num: i32 = fps.num.try_into()?;
        let den: i32 = fps.den.try_into()?;
        let extension = output
            .extension()
            .context("Output video needs a file extension")?
            .to_string_lossy();
        let parent = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temp = Builder::new()
            .suffix(&format!(".{extension}"))
            .tempfile_in(parent)?;
        let mut mux = format::output(temp.path())?;
        let codec =
            ffmpeg::encoder::find_by_name("libx265").context("libx265 encoder is unavailable")?;
        let global = mux.format().flags().contains(format::Flags::GLOBAL_HEADER);
        let mov = matches!(mux.format().name(), "mov" | "mp4");
        let mut stream = mux.add_stream(codec)?;
        let mut video = codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()?;
        video.set_width(width);
        video.set_height(height);
        video.set_aspect_ratio((1, 1));
        video.set_format(ffmpeg::format::Pixel::YUV444P);
        video.set_time_base((den, num));
        video.set_frame_rate(Some((num, den)));
        video.set_gop(((fps.to_fps()).round() as u32).max(1));
        if global {
            video.set_flags(codec::Flags::GLOBAL_HEADER);
        }
        let mut opts = Dictionary::new();
        opts.set("x265-params", "lossless=1:aq-mode=3");
        opts.set("profile", "main444-12");
        opts.set("crf", "18");
        let encoder = video.open_as_with(codec, opts)?;
        stream.set_time_base((den, num));
        stream.set_rate((num, den));
        stream.set_avg_frame_rate((num, den));
        stream.set_parameters(&encoder);
        if mov {
            unsafe {
                (*stream.parameters().as_mut_ptr()).codec_tag = u32::from_le_bytes(*b"hvc1");
            }
        }
        let audio = Audio::new(input, &mut mux)?;
        mux.write_header()?;
        let scaler = ffmpeg::software::scaling::Context::get(
            ffmpeg::format::Pixel::RGB24,
            width,
            height,
            ffmpeg::format::Pixel::YUV444P,
            width,
            height,
            ffmpeg::software::scaling::Flags::BICUBIC,
        )?;
        Ok(Self {
            output: output.to_path_buf(),
            mux: Some(mux),
            temp: Some(temp),
            encoder,
            scaler,
            audio,
            fps,
            frames: 0,
            first_video_packet: None,
            width,
            height,
        })
    }

    fn drain_video(&mut self) -> Result<()> {
        let mux = self.mux.as_mut().unwrap();
        let time_base = mux.stream(0).unwrap().time_base();
        loop {
            let mut packet = Packet::empty();
            match self.encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    packet.set_stream(0);
                    packet.rescale_ts(self.encoder.time_base(), time_base);
                    packet.write_interleaved(mux)?;
                    self.first_video_packet.get_or_insert_with(Instant::now);
                }
                Err(ffmpeg::Error::Eof) => return Ok(()),
                Err(e) if again(e) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    pub fn first_video_packet(&self) -> Option<Instant> {
        self.first_video_packet
    }

    pub fn write(&mut self, rgb: &frame::Video) -> Result<()> {
        ensure!(self.mux.is_some(), "Encoder already finished");
        ensure!(
            rgb.format() == ffmpeg::format::Pixel::RGB24
                && rgb.width() == self.width
                && rgb.height() == self.height,
            "Expected RGB24 frame at {}x{}",
            self.width,
            self.height
        );
        if let Some(audio) = self.audio.as_mut() {
            audio.until(self.mux.as_mut().unwrap(), self.frames, self.fps)?;
        }
        let mut yuv = frame::Video::empty();
        self.scaler.run(rgb, &mut yuv)?;
        yuv.set_pts(Some(self.frames));
        self.encoder.send_frame(&yuv)?;
        self.drain_video()?;
        self.frames = self
            .frames
            .checked_add(1)
            .context("Video frame count overflow")?;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        ensure!(self.mux.is_some(), "Encoder already finished");
        if let Some(audio) = self.audio.as_mut() {
            audio.flush(self.mux.as_mut().unwrap())?;
        }
        self.encoder.send_eof()?;
        self.drain_video()?;
        self.mux.as_mut().unwrap().write_trailer()?;
        drop(self.mux.take());
        self.temp.take().unwrap().persist(&self.output)?;
        Ok(())
    }
}

fn again(error: ffmpeg::Error) -> bool {
    error
        == ffmpeg::Error::Other {
            errno: ffmpeg::error::EAGAIN,
        }
}

fn later_than_frame(ts: i64, tb: Rational, frame: i64, fps: NTSC) -> bool {
    ts as i128 * tb.numerator() as i128 * fps.num as i128
        > frame as i128 * fps.den as i128 * tb.denominator() as i128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_pull_stops_at_future_video_timestamp() {
        let fps = NTSC {
            num: 24000,
            den: 1001,
        };
        assert!(!later_than_frame(0, Rational(1, 1000), 0, fps));
        assert!(!later_than_frame(41, Rational(1, 1000), 1, fps));
        assert!(later_than_frame(42, Rational(1, 1000), 1, fps));
        assert!(later_than_frame(2048, Rational(1, 48000), 1, fps));
    }
}
