use std::ops::Range;

pub struct Prng(pub u64);

impl Prng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn usize_range(&mut self, range: Range<usize>) -> usize {
        (self.next() % ((range.len() as i64 - 1).max(1) as u64) + range.start as u64) as usize
    }
    pub fn usize(&mut self, max: usize) -> usize {
        (self.next() % (max as u64)) as usize
    }
    pub fn f32(&mut self, r: Range<f32>) -> f32 {
        ((self.next() as u32) as f32 / u32::MAX as f32) * (r.end - r.start) + r.start
    }
    pub fn u8(&mut self, max: u8) -> u8 {
        (self.next() % (max as u64)) as u8
    }
}

