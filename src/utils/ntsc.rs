pub struct NTSC {
    pub num: u64,
    pub den: u64,
}

impl NTSC {
    pub fn new(num: &u64, den: &u64) -> Self {
        Self { num: *num, den: *den }
    }

    // Parse from string like "30000/1001"
    pub fn from_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() == 2 {
            if let (Ok(num), Ok(den)) = (parts[0].parse(), parts[1].parse()) {
                return Some(Self::new(&num, &den));
            }
        }
        None
    }

    // Create from strict fps like 24, 30, 60
    pub fn from_strict_fps(fps: &u64) -> Self {
        Self { num: *fps * 1000, den: 1001 }
    }

    // Convert to strict fps like 24, 30, 60
    pub fn as_strict_fps(&self) -> u64 {
        (self.num * 1001) / (self.den * 1000)
    }

    // Convert to floating point fps
    pub fn to_fps(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn clone(&self) -> Self {
        Self { num: self.num, den: self.den }
    }
}