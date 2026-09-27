require 'csv'

data = Dir.glob(File.join(__dir__, 'long-*-serial.log')).filter_map do |log|
  rows = File.foreach(log).filter_map do |line|
    match = line.match(/\A(\d+),([\d.]+),([\d.]+),([\d.]+)\s*\z/) ||
      line.match(/\Aframe=(\d+) preprocess_adapter_ms=([\d.]+) inference_sync_readback_ms=([\d.]+) total_ms=([\d.]+)/)
    match.captures.each_with_index.map { |value, i| i.zero? ? value.to_i : value.to_f } if match
  end
  unless rows.size == 120
    warn "excluded incomplete #{log}: #{rows.size}/120 frames"
    next
  end
  abort "unordered frames: #{log}" unless rows.each_with_index.all? { |row, i| row[0] == i + 1 }
  [log, rows]
end

summaries = data.map do |log, rows|
  CSV.open(log.sub(/\.log\z/, '.csv'), 'w') do |csv|
    csv << %w[frame preprocess_ms inference_ms total_ms]
    rows.each { |row| csv << row }
  end
  inference = rows.map { |row| row[2] }.sort
  [File.basename(log, '.log').delete_prefix('long-').delete_suffix('-serial'),
   rows.size, ((inference[59] + inference[60]) / 2).round(3),
   (120_000.0 / inference.sum).round(4),
   (120_000.0 / rows.sum { |row| row[3] }).round(4),
   inference.first, inference.last]
end

CSV.open(File.join(__dir__, 'continuous-ranking.csv'), 'w') do |csv|
  csv << %w[backend frames median_inference_ms inference_fps total_fps min_inference_ms max_inference_ms]
  summaries.sort_by { |row| -row[4] }.each { |row| csv << row }
end
