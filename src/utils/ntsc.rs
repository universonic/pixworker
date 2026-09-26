#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NTSC {
    pub num: u64,
    pub den: u64,
}

impl NTSC {
    pub fn new(num: &u64, den: &u64) -> Self {
        Self {
            num: *num,
            den: *den,
        }
    }

    // Parse from string like "30000/1001"
    pub fn from_string(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() == 2 {
            if let (Ok(num), Ok(den)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>())
                && num > 0
                && den > 0
            {
                return Some(Self::new(&num, &den));
            }
        }
        None
    }

    // Create from strict fps like 24, 30, 60
    pub fn from_strict_fps(fps: &u64) -> Self {
        Self { num: *fps, den: 1 }
    }

    // Convert to strict fps like 24, 30, 60
    pub fn as_strict_fps(&self) -> u64 {
        self.num / self.den
    }

    // Convert to floating point fps
    pub fn to_fps(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn scaled(&self, numerator: u64, denominator: u64) -> Option<Self> {
        if numerator == 0 || denominator == 0 {
            return None;
        }
        let mut num = self.num as u128 * numerator as u128;
        let mut den = self.den as u128 * denominator as u128;
        let (mut a, mut b) = (num, den);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        num /= a;
        den /= a;
        Some(Self {
            num: num.try_into().ok()?,
            den: den.try_into().ok()?,
        })
    }
}
