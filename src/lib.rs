//#![feature(trace_macros)]
//#![feature(portable_simd)]

#![deny(clippy::complexity)]
#![deny(clippy::correctness)]
#![deny(clippy::suspicious)]
#![deny(clippy::perf)]
#![deny(clippy::style)]
#![allow(clippy::new_without_default)]
#![warn(clippy::pedantic)]
#![allow(clippy::inline_always)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::wildcard_enum_match_arm)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::unnecessary_box_returns)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![deny(clippy::nursery)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::future_not_send)]
#![deny(clippy::suboptimal_flops)]

//#![deny(clippy::restrict)]

// #![deny(clippy::complexity)]
// #![deny(clippy::perf)]
// #![deny(clippy::style)]
// #![warn(clippy::pedantic)]
// #![warn(clippy::restriction)]
// #![allow(clippy::single_call_fn)]
// #![allow(clippy::arithmetic_side_effects)]
// #![allow(clippy::integer_division)]
// #![allow(clippy::absolute_paths)]
// #![allow(clippy::undocumented_unsafe_blocks)]
// #![allow(clippy::implicit_return)]
// #![allow(clippy::min_ident_chars)]
// #![allow(clippy::missing_docs_in_private_items)]
// #![allow(clippy::float_arithmetic)]
// #![allow(clippy::semicolon_outside_block)]
// #![allow(clippy::redundant_type_annotations)]
// #![allow(clippy::question_mark_used)]
// #![allow(clippy::as_conversions)]
// #![allow(clippy::unwrap_used)]
// #![allow(clippy::dbg_macro)]
// #![allow(clippy::print_stdout)]
// #![allow(clippy::use_debug)]
// #![allow(clippy::print_stderr)]
// #![allow(clippy::missing_inline_in_public_items)]
// #![allow(clippy::indexing_slicing)]
// #![allow(clippy::panic)]

use std::{
    array::from_fn,
    fmt::Debug,
    marker::PhantomData,
    mem::{replace, size_of, swap, ManuallyDrop},
    ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Range},
    time::Duration,
};

use no_panic::no_panic;
use wgpu::{util::DeviceExt, SurfaceError};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

// API
// cpu IR, gpu IR -> point buffer

const USE_GPU_COMPUTE: bool = false;

use prng::Prng;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod prng;
mod vertex;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

type IndexType = u16;

const CELLS_X: IndexType = 128;
const CELLS_Y: IndexType = 128;
const CELLS_MAX_DIM: IndexType = if CELLS_X > CELLS_Y { CELLS_X } else { CELLS_Y };
const CELLS: IndexType = CELLS_X * CELLS_Y;
const POINT_MAX_RADIUS: f32 = MIN_WIDTH / (CELLS_MAX_DIM as f32);

// in SIMULATION space, not view space
// exclusive upper bound
const MIN_X: f32 = 0.0;
const MAX_X: f32 = 1.0;

const MIN_Y: f32 = 0.0;
const MAX_Y: f32 = 1.0;

const WIDTH_X: f32 = MAX_X - MIN_X;
const WIDTH_Y: f32 = MAX_Y - MIN_Y;

const MIN_WIDTH: f32 = if WIDTH_X > WIDTH_Y { WIDTH_Y } else { WIDTH_X };
const K_FAC_BASE: f32 = 0.9;
#[cfg(feature = "nosimd")]
const K_FAC: f32 = K_FAC_BASE * 2.0 * 0.2;
#[cfg(not(feature = "nosimd"))]
const K_FAC: f32 = K_FAC_BASE * 2.0 * 0.2 * (1.0 / (1.0 - BETA));
const BETA: f32 = 0.3; //0.3;
const DT: f32 = 0.006; //0.008; // TODO: define max speed/max dt from cell size.
const NUM_POINTS_U: usize = NUM_POINTS as usize;
const CELLS_PLUS_1_U: usize = CELLS as usize + 1;
pub const NUM_POINTS: u32 = 32768; //16384; //8192; //1024; //16384; //8192; //2048; //16384;
const NUM_KINDS: usize = 8;
const NUM_KINDS2: usize = NUM_KINDS * NUM_KINDS;

fn debug_check_valid_pos(f: f32) -> bool {
    #[cfg(debug_assertions)]
    let r = (MIN_X..=MAX_X).contains(&f);
    #[cfg(not(debug_assertions))]
    let r = true;
    r
}

macro_rules! debug_assert_float {
    ($e:expr) => {{
        let f: f32 = $e;
        debug_assert!(!f.is_nan(), "{f}");
    }};
}
#[inline(always)]
fn scale_simulation_x(f: f32) -> f32 {
    MIN_X + WIDTH_X * f
}
#[inline(always)]
fn scale_simulation_y(f: f32) -> f32 {
    MIN_Y + WIDTH_Y * f
}
#[inline(always)]
fn inv_scale_simulation_x(f: f32) -> f32 {
    //(f - MIN_X) * (1.0 / WIDTH_X)
    // TODO: not respecting min_x, max_x
    f
}
#[inline(always)]
fn inv_scale_simulation_y(f: f32) -> f32 {
    //(f - MIN_Y) * (1.0 / WIDTH_Y)
    // TODO: not respecting min_x, max_x
    f
}
#[inline(always)]
fn mod_simulation_x(f: f32) -> f32 {
    (f - MIN_X).rem_euclid(WIDTH_X) + MIN_X
}
#[inline(always)]
fn mod_simulation_y(f: f32) -> f32 {
    (f - MIN_Y).rem_euclid(WIDTH_Y) + MIN_Y
}
fn reflect_sim_x(f: f32) -> f32 {
    // TODO: not respecting min_x, max_x
    // (((f.min(2.0-f).max(-f))-0.5)*0.999999)+0.5
    //f.min(2.0-f).max(-f)
    //f.min(5.0-f*4.0).max(-f*4.0)
    if f.is_nan() {
        eprintln!("nan x pos detected...");
        0.5
    } else if f < 0.0 {
        0.004
    } else if f > 1.0 {
        1.0 - 0.004
    } else {
        f
    }
}
fn reflect_sim_y(f: f32) -> f32 {
    // TODO: not respecting min_x, max_x
    // (((f.min(2.0-f).max(-f))-0.5)*0.999999)+0.5
    //f.min(2.0-f).max(-f)
    //f.min(5.0-f*4.0).max(-f*4.0)

    if f.is_nan() {
        eprintln!("nan y pos detected...");
        0.5
    } else if f < 0.0 {
        0.004
    } else if f > 1.0 {
        1.0 - 0.004
    } else {
        f
    }
}

#[inline(always)]
fn compute_cell(x: f32, y: f32) -> Cid {
    let x = inv_scale_simulation_x(x);
    let y = inv_scale_simulation_y(y);
    // x in 0_f32..1_f32
    // y in 0_f32..1_f32

    let cell_x = ((x * CELLS_X as f32) as IndexType).min(CELLS_X as IndexType - 1);
    let cell_y = ((y * CELLS_Y as f32) as IndexType).min(CELLS_Y as IndexType - 1);
    let cell = cell_x + cell_y * CELLS_X as IndexType;
    debug_assert!(
        cell < CELLS as <Cid as IndexWrapped>::IndexType,
        "{cell} >= CEllS, cell_x: {cell_x}, cell_y: {cell_y}, x: {x}, y: {y}"
    );
    Cid(cell)
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct EulerPoint {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}
unsafe impl bytemuck::Zeroable for EulerPoint {}
unsafe impl bytemuck::Pod for EulerPoint {}

// the randomly generated parts of the simulation
struct SimulationInputEuler {
    x: Pbox<f32>,
    y: Pbox<f32>,
    vx: Pbox<f32>,
    vy: Pbox<f32>,
    k: Pbox<u8>,
    relation_table: Box<[[f32; NUM_KINDS]; NUM_KINDS]>,
}
struct SimulationInputVerlet {
    x0: Pbox<f32>,
    y0: Pbox<f32>,
    x1: Pbox<f32>,
    y1: Pbox<f32>,
    k: Pbox<u8>,
    relation_table: Box<[[f32; NUM_KINDS]; NUM_KINDS]>,
}
#[derive(Copy, Clone)]
#[repr(C)]
struct Params {
    zoom_x: f32,
    zoom_y: f32,
    offset_x: f32,
    offset_y: f32,
}
unsafe impl bytemuck::Zeroable for Params {}
unsafe impl bytemuck::Pod for Params {}
impl Params {
    fn new() -> Self {
        Self {
            zoom_x: 1.0,
            zoom_y: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        }
    }
}
#[derive(Default)]
struct KeyState {
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    zoom_in: bool,
    zoom_out: bool,
}
impl KeyState {
    fn new() -> Self {
        Self::default()
    }
}
impl SimulationInputEuler {
    #[must_use]
    fn new() -> Self {
        Self::from_seed(
            17716589318410203473, ////238948293,
                                  //std::time::SystemTime::now()
                                  //    .duration_since(SystemTime::UNIX_EPOCH)
                                  //    .unwrap()
                                  //    .as_micros() as u64,
        )
    }
    #[must_use]
    fn from_seed(seed: u64) -> SimulationInputEuler {
        fn iter_alloc<T: std::fmt::Debug, const SIZE: usize, F: FnMut() -> T>(
            mut f: F,
        ) -> Box<[T; SIZE]> {
            (0..SIZE)
                .map(|_| f())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        }
        let mut rng = Prng(dbg!(seed));
        let xy_range = 0.0..1.0;
        let v_range = -0.1..0.1; //-1.0..1.0;

        SimulationInputEuler {
            x: iter_alloc(|| rng.f32(xy_range.clone())).into(),
            y: iter_alloc(|| rng.f32(xy_range.clone())).into(),
            vx: iter_alloc(|| rng.f32(v_range.clone())).into(),
            vy: iter_alloc(|| rng.f32(v_range.clone())).into(),
            k: iter_alloc(|| rng.u8(NUM_KINDS as u8)).into(),
            relation_table: Box::new(from_fn(|_| from_fn(|_| K_FAC * rng.f32(-1.0..1.0)))),
        }
    }
    fn verlet(self) -> SimulationInputVerlet {
        let x0 = self.x;
        let y0 = self.y;
        let mut x1 = self.vx;
        let mut y1 = self.vy;
        for (((x0, y0), x1), y1) in x0
            .iter()
            .zip(y0.iter())
            .zip(x1.iter_mut())
            .zip(y1.iter_mut())
        {
            *x1 = f32::mul_add(*x1, DT, *x0).rem_euclid(1.0);
            *y1 = f32::mul_add(*y1, DT, *y0).rem_euclid(1.0);
        }
        SimulationInputVerlet {
            x0,
            y0,
            x1,
            y1,
            k: self.k,
            relation_table: self.relation_table,
        }
    }
}
trait IndexWrapped {
    type IndexType;
}
macro_rules! index_wrap {
    ($name:ident, $index:ty, $doc:expr) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(Copy, Clone, Default, Debug)]
        struct $name($index);
        unsafe impl bytemuck::Zeroable for $name {}
        unsafe impl bytemuck::Pod for $name {}
        impl IndexWrapped for $name {
            type IndexType = $index;
        }
        impl From<$name> for usize {
            fn from(value: $name) -> usize {
                value.0 as _
            }
        }
        impl From<$name> for u32 {
            fn from(value: $name) -> u32 {
                value.0 as _
            }
        }
        impl From<usize> for $name {
            fn from(value: usize) -> $name {
                $name(value as _)
            }
        }
        impl AddAssign<$name> for $name {
            fn add_assign(&mut self, other: Self) {
                self.0 += other.0;
            }
        }
        impl Add<$name> for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }
        impl IntoGpu for $name {
            type GpuType = u32;
            fn into_gpu(self) -> Self::GpuType {
                self.0 as _
            }
        }
    };
}
index_wrap!(Pid, IndexType, "Point index");
index_wrap!(Cid, IndexType, "Cell index");
/// Packed "float" and 4 bit value, exploiting unused bits in a float to save some memory.
// #[repr(transparent)]
// #[derive(Copy, Clone, Default, Debug)]
// struct Packed(u32);
// impl Packed {
//     fn pack(float: f32, val: u8) -> u32 {
//         // float needs to be in the range 64.0_f32..128.0_f32
//         // 0b10111101000000000000000000000000
//         let mask: u32 = 0b0011_1100_0000_0000_0000_0000_0000_0000;
//         let shift = mask.trailing_zeros();
//         float.to_bits() | ((val as u32) << shift)
//     }
//     fn unpack(self) -> (f32, u8) {
//         // float needs to be in the range 64.0_f32..128.0_f32
//         // 0b10111101000000000000000000000000
//         let packed = self.0;
//         let mask: u32 = 0b0011_1100_0000_0000_0000_0000_0000_0000;
//         let shift = mask.trailing_zeros();
//         let float = f32::from_bits(packed & (!mask));
//         let val = ((packed & mask) >> shift) as u8;
//         (float, val)
//     }
// }

// lower_limit: 0.49
// upper_limit: 1.1
// always zero:    0b11000000000000000000000000000000 (2)
// always one:     0b00111110000000000000000000000000 (5)
// avaliable bits: 0b11111110000000000000000000000000 (7)
// unique float values: 9563013 (2^23 = 8388608), diff: 0.00000011920929
struct Packed(u32);
impl Packed {
    const _ASSERT: () = {
        // assert!(MIN_X > 0.49);
        // assert!(MAX_X < 1.1);
    };
    const MASK: u32 = 0b0011_1110_0000_0000_0000_0000_0000_0000;
    const SHIFT: u32 = Self::MASK.trailing_zeros();
    #[inline(always)]
    fn pack(float: f32, val: u8) -> u32 {
        (float.to_bits() & (!Self::MASK)) | ((val as u32) << Self::SHIFT)
    }
    #[inline(always)]
    fn unpack_float(packed: u32) -> f32 {
        f32::from_bits(packed | Self::MASK)
    }
    #[inline(always)]
    fn unpack_u8(packed: u32) -> u8 {
        ((packed & Self::MASK) >> Self::SHIFT) as u8
    }
    #[inline(always)]
    unsafe fn mm256_pack_preshifted(
        x: core::arch::x86_64::__m256,
        preshifted_k: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        use core::arch::x86_64::*;
        _mm256_or_si256(
            _mm256_and_si256(
                _mm256_set1_epi32((!Self::MASK) as i32),
                _mm256_castps_si256(x),
            ),
            preshifted_k,
        )
    }
    #[inline(always)]
    unsafe fn mm256_unpack_preshifted_k(
        combined: core::arch::x86_64::__m256i,
    ) -> core::arch::x86_64::__m256i {
        use core::arch::x86_64::*;
        _mm256_and_si256(_mm256_set1_epi32(Self::MASK as i32), combined)
    }
    #[inline(always)]
    unsafe fn mm256_unpack_k(combined: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
        use core::arch::x86_64::*;
        _mm256_srli_epi32(
            _mm256_and_si256(_mm256_set1_epi32(Self::MASK as i32), combined),
            Self::SHIFT as i32,
        )
    }
    #[inline(always)]
    unsafe fn mm256_unpack_x(combined: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
        use core::arch::x86_64::*;
        _mm256_or_si256(_mm256_set1_epi32(Self::MASK as i32), combined)
    }
}

#[derive(Debug)]
struct Tb<I, T, const SIZE: usize>(Box<[T; SIZE]>, PhantomData<I>);
impl<I, T, const SIZE: usize> From<Box<[T; SIZE]>> for Tb<I, T, SIZE> {
    fn from(value: Box<[T; SIZE]>) -> Self {
        Self(value, PhantomData)
    }
}

impl<I, T, const SIZE: usize> Deref for Tb<I, T, SIZE> {
    type Target = Box<[T; SIZE]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<I, T, const SIZE: usize> DerefMut for Tb<I, T, SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<I: Into<usize>, T, const SIZE: usize> Index<I> for Tb<I, T, SIZE> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        //&self.0[index.into()]
        unsafe { self.0.get_unchecked(index.into()) }
    }
}
impl<I: Into<usize>, T, const SIZE: usize> IndexMut<I> for Tb<I, T, SIZE> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        //&mut self.0[index.into()]
        unsafe { self.0.get_unchecked_mut(index.into()) }
    }
}
type Pbox<T> = Tb<Pid, T, NUM_POINTS_U>;
type Cbox<T> = Tb<Cid, T, CELLS_PLUS_1_U>;

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread;

struct SimThreadController {
    recv_data: Receiver<Vec<GpuPoint>>,
    counter: Arc<AtomicUsize>,
}
impl SimThreadController {
    fn get_state(&mut self, v: &mut Vec<GpuPoint>) {
        *v = self.recv_data.recv().unwrap();
    }
    fn try_get_state(&mut self, v: &mut Vec<GpuPoint>) {
        if let Ok(r) = self.recv_data.try_recv() {
            self.counter.fetch_sub(1, Relaxed);
            *v = r;
        } else {
            dbg!();
        }
    }
    fn new() -> Self {
        let (send, recv) = channel();
        let counter = Arc::new(AtomicUsize::new(0));
        let their_counter = counter.clone();
        thread::spawn(|| {
            let send = send;
            let counter = their_counter;
            let mut simulation_state = CpuSimulationState::new();
            let mut delta = Instant::now();
            let mut updates = 0;
            let mut cycles_last = unsafe { core::arch::x86_64::_rdtsc() };
            loop {
                for _ in 0..1 {
                    simulation_state.update();
                    updates += 1;
                }
                if counter.load(Relaxed) < 2 {
                    let mut v = Vec::new();
                    simulation_state.serialize_gpu(&mut v);
                    counter.fetch_add(1, Relaxed);
                    if send.send(v).is_err() {
                        break;
                    };
                } else {
                    //std::thread::sleep(Duration::from_millis(10));
                }
                let elapsed = delta.elapsed();
                if elapsed > Duration::from_secs(1) {
                    let ups = updates as f64 / elapsed.as_secs_f64();
                    let cycles_now = unsafe { core::arch::x86_64::_rdtsc() };
                    let cycles_elapsed = cycles_now - replace(&mut cycles_last, cycles_now);
                    let cycles_per_point_update =
                        cycles_elapsed as f64 / (updates as f64 * NUM_POINTS as f64);
                    println!("ups: {ups}, cppu: {cycles_per_point_update}");
                    updates = 0;
                    delta = Instant::now();
                }
            }
        });

        Self {
            recv_data: recv,
            counter,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GpuPoint {
    x: f32,
    y: f32,
    k: u32,
    _unused: u32,
}
unsafe impl bytemuck::Zeroable for GpuPoint {}
unsafe impl bytemuck::Pod for GpuPoint {}

type UnsafePointVec<T> = UnsafeVec<T, NUM_POINTS_U>;

struct UnsafeVec<T, const SIZE: usize> {
    inner: Box<[T; SIZE]>,
}
impl<T, const SIZE: usize> UnsafeVec<T, SIZE> {
    #[inline(always)]
    unsafe fn end_ptr_mut(&mut self, len: usize) -> *mut T {
        self.inner.as_mut_ptr().add(len)
    }
}

impl<T, const SIZE: usize> From<Box<[T; SIZE]>> for UnsafeVec<T, SIZE> {
    fn from(inner: Box<[T; SIZE]>) -> Self {
        Self { inner }
    }
}

impl<T, const SIZE: usize> Deref for UnsafeVec<T, SIZE> {
    type Target = Box<[T; SIZE]>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl<T, const SIZE: usize> DerefMut for UnsafeVec<T, SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

fn calloc_counted<T: Default + Debug + bytemuck::Zeroable + bytemuck::Pod, const SIZE: usize, O>(
    byte_counter: &mut usize,
) -> O
where
    O: From<Box<[T; SIZE]>>,
{
    //let b: Box<[T; SIZE]> = (0..SIZE)
    //    .map(|_| T::default())
    //    .collect::<Vec<T>>()
    //    .try_into()
    //    .unwrap();
    *byte_counter += size_of::<T>() * SIZE;
    calloc()
}

fn calloc<T: Default + Debug + bytemuck::Zeroable + bytemuck::Pod, const SIZE: usize, O>() -> O
where
    O: From<Box<[T; SIZE]>>,
{
    let b = aligned_alloc_64();
    let alignment = 1 << (b.as_ptr() as usize).trailing_zeros();
    println!(
        "ALIGNMENT: {}, LEN: {} {}",
        alignment,
        b.len(),
        if alignment >= 64 { "(OK)" } else { panic!() }
    );
    O::from(b)
}

fn aligned_alloc_64<T: Default + bytemuck::Zeroable + bytemuck::Pod, const SIZE: usize>(
) -> Box<[T; SIZE]> {
    #[repr(align(64), C)]
    struct AlignedBytes([u8; 64]);
    impl Default for AlignedBytes {
        fn default() -> Self {
            Self([0; 64])
        }
    }
    aligned_alloc::<AlignedBytes, T, SIZE>()
}

fn aligned_alloc_4096<T: Default + bytemuck::Zeroable + bytemuck::Pod, const SIZE: usize>(
) -> Box<[T; SIZE]> {
    #[repr(align(4096), C)]
    struct AlignedBytes([u8; 4096]);
    impl Default for AlignedBytes {
        fn default() -> Self {
            Self([0; 4096])
        }
    }
    aligned_alloc::<AlignedBytes, T, SIZE>()
}

fn aligned_alloc<S: Default, T: Default + bytemuck::Zeroable + bytemuck::Pod, const SIZE: usize>(
) -> Box<[T; SIZE]> {
    let mut v = ManuallyDrop::new(
        (0..((SIZE * size_of::<T>()) / size_of::<S>()))
            .map(|_| S::default())
            .collect::<Vec<S>>(),
    );
    v.shrink_to_fit();
    unsafe { Box::from_raw(v.as_mut_ptr() as *mut [T; SIZE]) }
}

trait GpuRepresentable: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {}

impl GpuRepresentable for f32 {}
impl GpuRepresentable for u32 {}

trait IntoGpu: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {
    type GpuType: GpuRepresentable;
    fn into_gpu(self) -> Self::GpuType;
}
macro_rules! impl_into_gpu {
    ($from:ty, $into:ty) => {
        impl IntoGpu for $from {
            type GpuType = $into;
            fn into_gpu(self) -> Self::GpuType {
                self as _
            }
        }
    };
}
impl_into_gpu!(u8, u32);
impl_into_gpu!(u16, u32);
impl_into_gpu!(u32, u32);
impl_into_gpu!(f32, f32);

// struct TypedBuffer<T: GpuRepresentable> {
//     buffer: wgpu::Buffer,
//     ty: PhantomData<T>,
//     count: u32,
// }
//
// struct Pipelines {
//     serialize: wgpu::ComputePipeline,
//     update: wgpu::ComputePipeline,
//     ppa_256: wgpu::ComputePipeline,
//     ppa_full: wgpu::ComputePipeline,
// }

/*struct GpuSimulationState {
  x0: TypedBuffer<f32>, // f32
  y0: TypedBuffer<f32>, // f32
  x1: TypedBuffer<f32>, // f32
  y1: TypedBuffer<f32>, // f32
  k: TypedBuffer<u32>,  // u8 -> u32

  x0_tmp: TypedBuffer<f32>, // f32
  y0_tmp: TypedBuffer<f32>, // f32
  x1_tmp: TypedBuffer<f32>, // f32
  y1_tmp: TypedBuffer<f32>, // f32
  k_tmp: TypedBuffer<u32>,  // u8 -> u32

  point_cell: TypedBuffer<u32>, // Cid -> u32

  point_cell_offset: TypedBuffer<u32>, // Pid -> u32

  cell_count: TypedBuffer<u32>, // Pid -> u32
  cell_index: TypedBuffer<u32>, // Pid -> u32

  relation_table: TypedBuffer<f32>, // f32
                                    //
                                    // update points from surrounding data, compute cell
                                    // increment counts
                                    //update_pipeline: wgpu::ComputePipeline,
                                    //update_bind_group: wgpu::BindGroup,

                                    // increment count for cell, shared memory too small though
                                    // count_pipeline: wgpu::ComputePipeline,
                                    // count_bind_group: wgpu::BindGroup,

                                    // perform ppa, although shared memory is small
                                    // ppa_pipeline: wgpu::ComputePipeline,
                                    // ppa_bind_group: wgpu::BindGroup,

                                    // insert into calculated offsets. merge with previous?
                                    // insert_pipeline: wgpu::ComputePipeline,
                                    // insert_bind_group: wgpu::BindGroup,
                                    // point_serialize_pipeline: wgpu::ComputePipeline,
                                    // point_serialize_bind_group: wgpu::BindGroup,
                                    pipelines: Pipelines,
                                    //serialize_pipeline: wgpu::ComputePipeline,
                                    //update_pipeline: wgpu::ComputePipeline,
                                    bind_groups: [wgpu::BindGroup; 2],
                                    }
                                    impl GpuSimulationState {
                                    const WORKGROUP_SIZE: u16 = 256; // the max
                                    fn from_simulation_state(
                                    device: &wgpu::Device,
                                    point_buffer: &wgpu::Buffer,
                                    state: &CpuSimulationState,
                                    ) -> Self {
// TODO: better allocation strategy?
fn galloc<T: IntoGpu>(device: &wgpu::Device, content: &[T]) -> TypedBuffer<T::GpuType> {
let mapped_data: Vec<_> = content.iter().copied().map(T::into_gpu).collect();
let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
label: None,
contents: bytemuck::cast_slice(&mapped_data),
usage: wgpu::BufferUsages::STORAGE,
});
TypedBuffer {
buffer,
ty: PhantomData,
count: content.len().try_into().unwrap(),
}
}

let x0 = galloc(device, state.x0.0.as_slice());
let y0 = galloc(device, state.y0.0.as_slice());
let x1 = galloc(device, state.x1.0.as_slice());
let y1 = galloc(device, state.y1.0.as_slice());
let k = galloc(device, state.k.0.as_slice());
let x0_tmp = galloc(device, state.x0_tmp.0.as_slice());
let y0_tmp = galloc(device, state.y0_tmp.0.as_slice());
let x1_tmp = galloc(device, state.x1_tmp.0.as_slice());
let y1_tmp = galloc(device, state.y1_tmp.0.as_slice());
let k_tmp = galloc(device, state.k_tmp.0.as_slice());
let point_cell = galloc(device, state.point_cell.0.as_slice());
let point_cell_offset = galloc(device, state.point_cell_offset.0.as_slice());
let cell_count = galloc(device, state.cell_count.0.as_slice());
let cell_index = galloc(device, state.cell_index.0.as_slice());
let relation_table = galloc(
    device,
    bytemuck::cast_slice::<_, f32>(state.relation_table.as_slice()),
    );

#[rustfmt::skip]
let gpu_vars = [
    (x0.count, [&x0.buffer, &x0.buffer], "x0", "f32"),
    (y0.count, [&y0.buffer, &y0.buffer], "y0", "f32"),
    (x1.count, [&x1.buffer, &x1.buffer], "x1", "f32"),
    (y1.count, [&y1.buffer, &y1.buffer], "y1", "f32"),
    (k.count, [&k.buffer, &k.buffer], "k", "u32"),
    (x0_tmp.count, [&x0_tmp.buffer, &x0_tmp.buffer], "x0_tmp", "f32"),
    (y0_tmp.count, [&y0_tmp.buffer, &y0_tmp.buffer], "y0_tmp", "f32"),
    (x1_tmp.count, [&x1_tmp.buffer, &x1_tmp.buffer], "x1_tmp", "f32"),
    (y1_tmp.count, [&y1_tmp.buffer, &y1_tmp.buffer], "y1_tmp", "f32"),
    (k_tmp.count, [&k_tmp.buffer, &k_tmp.buffer], "k_tmp", "u32"),
    (point_cell.count, [&point_cell.buffer, &point_cell.buffer], "point_cell", "u32"),
    (point_cell_offset.count, [&point_cell_offset.buffer, &point_cell_offset.buffer], "point_cell_offset", "u32"),
    (cell_count.count, [&cell_count.buffer, &cell_count.buffer], "cell_count", "atomic<u32>"),
    (cell_index.count, [&cell_index.buffer, &cell_index.buffer], "cell_index", "u32"),
    (relation_table.count, [&relation_table.buffer, &relation_table.buffer], "relation_table", "f32"),
    (NUM_POINTS, [&point_buffer, &point_buffer], "points", "GpuPoint"),
];

    let layout_entries = gpu_vars
    .iter()
.enumerate()
    .map(|(i, (_, b, _, _))| wgpu::BindGroupLayoutEntry {
        binding: i as _,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(
                b.iter().map(|b| b.size()).all_equal_value().unwrap(),
                ),
        },
        count: None,
    })
.collect::<Vec<_>>();

let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
    label: None,
    entries: &layout_entries,
});

let bind_entries: [_; 2] = from_fn(|e| {
    gpu_vars
        .iter()
        .enumerate()
        .map(|(i, (_, b, _, _))| wgpu::BindGroupEntry {
            binding: i as _,
            resource: b[e].as_entire_binding(),
        })
    .collect::<Vec<_>>()
});

let bind_groups = bind_entries.map(|bind_entries| {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_layout,
        entries: &bind_entries,
    })
});

let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: None,
    bind_group_layouts: &[&bind_layout],
    push_constant_ranges: &[],
});

    let wgpu_declarations: String = gpu_vars
    .iter()
.enumerate()
    .map(|(i, (count, _, name, ty))| {
        format!(
            "@group(0) @binding({i}) var<storage, read_write> {name}: array<{ty}, {}>;\n",
            count
            )
    })
.collect();

    let source = std::fs::read_to_string("src/compute.wgsl")
.unwrap()
    .replace("NUM_POINTS", &format!("{NUM_POINTS}"))
    .replace("MIN_WIDTH", &format!("{MIN_WIDTH}"))
    .replace("NUM_KINDS", &format!("{NUM_KINDS}"))
    .replace("CELLS_Y", &format!("{CELLS_Y}"))
    .replace("CELLS_X", &format!("{CELLS_X}"))
    .replace("CELLS", &format!("{CELLS}"))
    .replace("POINT_MAX_RADIUS", &format!("{POINT_MAX_RADIUS}"))
    .replace("BETA", &format!("{BETA}"))
    .replace("DT", &format!("{DT}"))
    .replace("MIN_Y", &format!("{MIN_Y:?}"))
    .replace("MIN_X", &format!("{MIN_X:?}"))
    .replace("MAX_Y", &format!("{MAX_Y:?}"))
    .replace("MAX_X", &format!("{MAX_X:?}"))
    .replace(
        "PPA_BATCHES",
        &format!("{:?}", NUM_POINTS.checked_div(256).unwrap()),
        )
    .replace("WORKGROUP_SIZE", &format!("{}", Self::WORKGROUP_SIZE));

    assert!(NUM_POINTS <= 32768, "ppa needs to be updated");

    let source = format!("{}\n{}", wgpu_declarations, source);
    println!("SHADER SOURCE: \n\n {}\n\n", &source);

    let module_descriptor = wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::from(source)),
    };

let module = device.create_shader_module(module_descriptor);

let pipe = |entry_point| {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point,
    })
};

let pipelines = Pipelines {
    serialize: pipe("serialize"),
    update: pipe("update"),
    ppa_256: pipe("exclusive_ppa256"),
    ppa_full: pipe("full_ppa"),
};

Self {
    x0,
    y0,
    x1,
    y1,
    k,
    x0_tmp,
    y0_tmp,
    x1_tmp,
    y1_tmp,
    k_tmp,
    point_cell,
    point_cell_offset,
    cell_count,
    cell_index,
    relation_table,
    bind_groups,
    pipelines,
}
}
fn update<'this, 'pass>(&'this self, pass: &mut wgpu::ComputePass<'pass>)
where
'this: 'pass,
{
    pass.set_bind_group(0, &self.bind_groups[0], &[]);
    self.update_i(pass);
    pass.set_bind_group(0, &self.bind_groups[1], &[]);
    self.update_i(pass);
}

fn update_i<'this, 'pass>(&'this self, pass: &mut wgpu::ComputePass<'pass>)
where
'this: 'pass,
{
    pass.set_pipeline(&self.pipelines.update);
    Self::dispatch(Self::WORKGROUP_SIZE, NUM_POINTS, pass);
    pass.set_pipeline(&self.pipelines.ppa_256);
    Self::dispatch(256, u32::try_from(CELLS).unwrap(), pass);
    pass.set_pipeline(&self.pipelines.ppa_full);
    let final_size = u32::try_from(CELLS).unwrap().checked_div(256).unwrap();
    assert!(final_size <= 256);
    Self::dispatch(256, final_size, pass);
}
fn serialize_gpu<'this, 'pass>(&'this self, pass: &mut wgpu::ComputePass<'pass>)
where
'this: 'pass,
{
    pass.set_bind_group(0, &self.bind_groups[0], &[]);

    pass.set_pipeline(&self.pipelines.serialize);
    Self::dispatch(Self::WORKGROUP_SIZE, NUM_POINTS, pass);
}
fn dispatch<'a>(
    workgroup_size: impl TryInto<u32>,
    total_size: impl TryInto<u32>,
    pass: &mut wgpu::ComputePass<'a>,
    ) {
    let (workgroup_size, total_size) = (
        workgroup_size.try_into().unwrap_or_else(|_| panic!()),
        total_size.try_into().unwrap_or_else(|_| panic!()),
        );
    pass.dispatch_workgroups(
        total_size
        .checked_div(workgroup_size)
        .unwrap_or_else(|| panic!()),
        1,
        1,
        );
}
fn dispatch_size(workgroup_size: impl TryInto<u32>, total_size: impl TryInto<u32>) -> u32 {
    let (workgroup_size, total_size) = (
        workgroup_size.try_into().unwrap_or_else(|_| panic!()),
        total_size.try_into().unwrap_or_else(|_| panic!()),
        );
    total_size.div_ceil(workgroup_size)
}
}*/

/// Automatically selects appropriate fused multiply add/sub intrinsic.
/// will perform simple add/sub/mul if surrounded in parenthesis.
/// function calls are allowed, but their arguments are invisible to the macro
#[rustfmt::skip]
macro_rules! fma {

    ( set1 ( $a:expr , $b:expr ) )      => { core::arch::x86_64::_mm256_set1_ps($a, $b) };
    ( min ( $a:expr , $b:expr ) )       => { core::arch::x86_64::_mm256_min_ps($a, $b) };
    ( max ( $a:expr , $b:expr ) )       => { core::arch::x86_64::_mm256_max_ps($a, $b) };


    (           $a:tt * $b:tt + $c:expr ) => { core::arch::x86_64::_mm256_fmadd_ps($a, $b, $c) };
    (   $c:tt + $a:tt * $b:expr         ) => { core::arch::x86_64::_mm256_fmadd_ps($a, $b, $c) };

    (           $a:tt * $b:tt - $c:expr ) => { core::arch::x86_64::_mm256_fmsum_ps($a, $b, $c) };
    ( - $c:tt + $a:tt * $b:expr         ) => { core::arch::x86_64::_mm256_fmsum_ps($a, $b, $c) };

    (         - $a:tt * $b:tt + $c:expr ) => { core::arch::x86_64::_mm256_fnmadd_ps($a, $b, $c) };
    (   $c:tt - $a:tt * $b:expr         ) => { core::arch::x86_64::_mm256_fnmadd_ps($a, $b, $c) };

    (         - $a:tt * $b:tt - $c:expr ) => { core::arch::x86_64::_mm256_fnmsum_ps($a, $b, $c) };
    ( - $c:tt - $a:tt * $b:expr         ) => { core::arch::x86_64::_mm256_fnmsum_ps($a, $b, $c) };

    ( $a:tt - $b:expr )           => { core::arch::x86_64::_mm256_sub_ps($a, $b) };
    ( -$b:tt + $a:expr)           => { core::arch::x86_64::_mm256_sub_ps($a, $b) };
    ( $a:tt + $b:expr )           => { core::arch::x86_64::_mm256_add_ps($a, $b) };
    ( $a:tt * $b:expr )           => { core::arch::x86_64::_mm256_mul_ps($a, $b) };

    ( $a:tt & $b:tt)                => { core::arch::x86_64::_mm256_and_ps($a, $b) };
    ( ! ( $a:tt ) & $b:tt)          => { core::arch::x86_64::_mm256_andnot_ps($a, $b) };
    ( $a:tt | $b:tt)                => { core::arch::x86_64::_mm256_or_ps($a, $b) };
    ( $a:tt != $b:tt )              => { core::arch::x86_64::_mm256_cmp_ps($a, $b, _CMP_NEQ_OQ) };
    // ( $f:ident ( $a:tt ) )          => { $f($a) };
    // ( $f:ident ( ) )                => { $f() };
    ( 0.0 )                         => { core::arch::x86_64::_mm256_setzero_ps() };
    ( 0 )                           => { core::arch::x86_64::_mm256_setzero_ps() };
}

fn test() {
    macro_rules! multi_alloc {
        ($a:ident) => {
            let $a = calloc();
        };
    }

    macro_rules! find_min {
    // Base case:
    ($x:expr) => ($x);
    // `$x` followed by at least one `$y,`
    ($x:expr, $($y:expr),+) => (
        // Call `find_min!` on the tail `$y`
        std::cmp::min($x, find_min!($($y),+))
    )
}

    // let x0;
    // let y0;
    // let x1;
    // let y1;
    // let k;
    // let x0_tmp;
    // let y0_tmp;
    // let x1_tmp;
    // let y1_tmp;
    // let k_tmp;
    // let point_cell;
    // let point_cell_offset;
    // let cell_count;
    // let cell_index;
    // let relation_table;

    macro_rules! multi_alloc {
        ($a:ident, $($b:ident),+) => {
            multi_alloc!($a);
            multi_alloc!($($b),+);
        };
        ($a:ident) => {
            let $a = calloc();
        };
    }

    // multi_alloc!(
    //     x0,
    //     y0,
    //     x1,
    //     y1,
    //     k,
    //     x0_tmp,
    //     y0_tmp,
    //     x1_tmp,
    //     y1_tmp,
    //     k_tmp,
    //     point_cell,
    //     point_cell_offset,
    //     cell_count,
    //     cell_index,
    //     relation_table
    // );
    //    macro_rules! my_macro {
    //     (struct $name:ident {
    //         $($field_name:ident: $field_type:ty,)*
    //     }) => {
    //         struct $name {
    //             $($field_name: $field_type,)*
    //         }
    //
    //         impl $name {
    //             // This is purely an example—not a good one.
    //             fn get_field_names() -> Vec<&'static str> {
    //                 vec![$(stringify!($field_name)),*]
    //             }
    //         }
    //     }
    // }
    //
    // my_macro! {
    //     struct S {
    //         a: String,
    //         b: String,
    //     }
    // }
    {
        // ((SIZE * size_of::<T>()) / size_of::<S>())

        // let mut v = (0..(SIZE.div_ceil(size_of::<AlignedBytes>())))
        //     .map(|_| AlignedBytes([0;4096]))
        //     .collect::<Vec<AlignedBytes>>();
        // v.shrink_to_fit();
        // v.into_boxed_slice();
    }
    #[no_panic]

    fn tt(a: &mut [u64; 4]) {
        let b: &mut [u64; 8] = bytemuck::cast_mut(a);
    }

    macro_rules! make_state {
        ($name:ident $borrowed_name:ident { $($field_name:ident : [ $t:ty ; $e:expr],)* })=> {


            struct $borrowed_name < 'a > {
                $($field_name : & 'a mut [$t ; $e],)*
            }
            impl<'a> $borrowed_name <'a> {
                fn new(data: &'a mut [AlignedBytes; ALLOC_SIZE]) -> $borrowed_name <'a> {
                    let data: &mut [u8] = bytemuck::cast_slice_mut(data.as_mut_slice());
                    $(
                        let alloc_size = $e * std::mem::size_of::<$t>();
                        let $field_name: &mut [u8] = &mut [];
                        let $field_name: &mut [$t] = bytemuck::cast_slice_mut($field_name);
                        let $field_name: &mut [$t; $e] = $field_name.try_into().unwrap();
                    )*

                    $borrowed_name {
                        $($field_name,)*
                    }
                }
            }
            const ALIGN: usize = 128;
            const ALLOC_SIZE_BYTES: usize = calc_alloc_size();
            const ALLOC_SIZE: usize = ALLOC_SIZE_BYTES.div_ceil(size_of::<AlignedBytes>());
            const fn calc_alloc_size() -> usize {
                let mut offset = 0;
                $(offset = (offset + $e * std::mem::size_of::<$t>()).next_multiple_of(ALIGN);)*
                offset
            }


            unsafe impl bytemuck::Zeroable for AlignedBytes {}
            unsafe impl bytemuck::Pod for AlignedBytes {}
            #[derive(Copy, Clone)]
            #[repr(align(4096), C)]
            struct AlignedBytes([u8; 4096]);
            struct $name (Box<[AlignedBytes; ALLOC_SIZE]>);
            impl $name {
                fn new() -> Self {
                    let Ok(v) = (0..ALLOC_SIZE).map(|_| AlignedBytes([0; 4096]))
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                        .try_into() else { panic!() };
                    Self( v )
                }
            }
        }
    }

    make_state!(
        State BorrowedState {
            x0: [f32; NUM_POINTS_U],
            y0: [f32; NUM_POINTS_U],
            x1: [f32; NUM_POINTS_U],
            y1: [f32; NUM_POINTS_U],
            k: [u8; NUM_POINTS_U],
            x0_tmp: [f32; NUM_POINTS_U],
            y0_tmp: [f32; NUM_POINTS_U],
            x1_tmp: [f32; NUM_POINTS_U],
            y1_tmp: [f32; NUM_POINTS_U],
            k_tmp: [u8; NUM_POINTS_U],
            point_cell: [Cid; NUM_POINTS_U],
            point_cell_offset: [Pid; NUM_POINTS_U],
            cell_count: [Pid; CELLS_PLUS_1_U],
            cell_index: [Pid; CELLS_PLUS_1_U],
            relation_table: [f32; NUM_KINDS2],
        }
    );

    // struct BorrowedState<'a> {
    //     x0: &'a mut [f32; NUM_POINTS_U],
    //     y0: &'a mut [f32; NUM_POINTS_U],
    //     x1: &'a mut [f32; NUM_POINTS_U],
    //     y1: &'a mut [f32; NUM_POINTS_U],
    //     k: &'a mut [u8; NUM_POINTS_U],
    //     x0_tmp: &'a mut [f32; NUM_POINTS_U],
    //     y0_tmp: &'a mut [f32; NUM_POINTS_U],
    //     x1_tmp: &'a mut [f32; NUM_POINTS_U],
    //     y1_tmp: &'a mut [f32; NUM_POINTS_U],
    //     k_tmp: &'a mut [u8; NUM_POINTS_U],
    //     point_cell: &'a mut [Cid; NUM_POINTS_U],
    //     point_cell_offset: &'a mut [Pid; NUM_POINTS_U],
    //     cell_count: &'a mut [Pid; CELLS_PLUS_1_U],
    //     cell_index: &'a mut [Pid; CELLS_PLUS_1_U],
    //     relation_table: &'a mut [f32; NUM_KINDS2],
    // }

    //macro_rules! make_state {
    //    // ($field:ident) => {
    //    //
    //    //     //let $field: &mut [$t ; $size];

    //    // };

    //    ($field:ident , $($fieldd:ident),*) => {

    //        //make_state!($field : [$t ; $size ]);

    //    };
    //}

    // trace_macros!(true);
    // macro_rules! make_state_inner {
    //     ($l:lifetime, $i:ident, $t:ty, $e:expr) => {
    //         $i: & $l mut [ $t ; $e ],
    //     };
    //     ($l:lifetime, $ii:ident, $tt:ty, $ee:expr, $($i:ident, $t:ty, $e:expr),+) => {
    //         make_state_inner!($l, $ii, $tt, $ee)
    //         make_state_inner!($l, $($i, $t, $e),+)
    //     };
    // }

    // macro_rules! make_state {
    //     //($x:ident) => {};
    //     ($l:lifetime $name:ident $($i:ident : [ $t:ty ; $e:expr ]),+) => {
    //         struct $name {
    //             make_state_inner!($l, $($i, $t, $e),+)
    //         }
    //     };
    // }

    // make_state!(
    //     'a BorrowedThing
    //     x0: [f32; NUM_POINTS_U],
    //     y0: [f32; NUM_POINTS_U],
    //     x1: [f32; NUM_POINTS_U],
    //     y1: [f32; NUM_POINTS_U],
    //     k: [u8; NUM_POINTS_U],
    //     x0_tmp: [f32; NUM_POINTS_U],
    //     y0_tmp: [f32; NUM_POINTS_U],
    //     x1_tmp: [f32; NUM_POINTS_U],
    //     y1_tmp: [f32; NUM_POINTS_U],
    //     k_tmp: [u8; NUM_POINTS_U],
    //     point_cell: [Cid; NUM_POINTS_U],
    //     point_cell_offset: [Pid; NUM_POINTS_U],
    //     cell_count: [Pid; CELLS_PLUS_1_U],
    //     cell_index: [Pid; CELLS_PLUS_1_U],
    //     relation_table: [f32; NUM_KINDS]
    // );

    // trace_macros!(false);

    // x0 = calloc();
    // y0 = calloc();
    // x1 = calloc();
    // y1 = calloc();
    // k = calloc();
    // x0_tmp = calloc();
    // y0_tmp = calloc();
    // x1_tmp = calloc();
    // y1_tmp = calloc();
    // k_tmp = calloc();
    // point_cell = calloc();
    // point_cell_offset = calloc();
    // cell_count = calloc();
    // cell_index = calloc();
    // relation_table = calloc();

    // CpuSimulationState {
    //     x0,
    //     y0,
    //     x1,
    //     y1,
    //     k,
    //     x0_tmp,
    //     y0_tmp,
    //     x1_tmp,
    //     y1_tmp,
    //     k_tmp,
    //     point_cell,
    //     point_cell_offset,
    //     cell_count,
    //     cell_index,
    //     relation_table,
    // };
}

/// Entire state of the simulation
pub struct CpuSimulationState {
    x0: Pbox<f32>,
    y0: Pbox<f32>,
    x1: Pbox<f32>,
    y1: Pbox<f32>,
    // TODO: merge with f32 unused bits
    k: Pbox<u8>,

    x0_tmp: Pbox<f32>,
    y0_tmp: Pbox<f32>,
    x1_tmp: Pbox<f32>,
    y1_tmp: Pbox<f32>,
    // TODO: merge with f32 unused bits
    k_tmp: Pbox<u8>,

    // TODO: pack (xxxxxxxxyyyyyyyy) in single array.
    point_cell: Pbox<Cid>,

    point_cell_offset: Pbox<Pid>,

    cell_count: Cbox<Pid>,
    cell_index: Cbox<Pid>,

    relation_table: Box<[[f32; NUM_KINDS]; NUM_KINDS]>,
}
impl CpuSimulationState {
    #[must_use]
    pub fn new() -> Self {
        Self::from_verlet(SimulationInputEuler::new().verlet())
    }
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self::from_verlet(SimulationInputEuler::from_seed(seed).verlet())
    }
    #[must_use]
    fn from_verlet(input: SimulationInputVerlet) -> Self {
        let SimulationInputVerlet {
            mut x0,
            mut y0,
            mut x1,
            mut y1,
            k,
            relation_table,
        } = input;
        x0.iter_mut().for_each(|e| *e = scale_simulation_x(*e));
        y0.iter_mut().for_each(|e| *e = scale_simulation_y(*e));
        x1.iter_mut().for_each(|e| *e = scale_simulation_x(*e));
        y1.iter_mut().for_each(|e| *e = scale_simulation_y(*e));
        let mut counter = 0;
        let mut this = Self {
            x0: calloc_counted(&mut counter),
            y0: calloc_counted(&mut counter),
            x1: calloc_counted(&mut counter),
            y1: calloc_counted(&mut counter),
            k: calloc_counted(&mut counter),
            x0_tmp: calloc_counted(&mut counter),
            y0_tmp: calloc_counted(&mut counter),
            x1_tmp: calloc_counted(&mut counter),
            y1_tmp: calloc_counted(&mut counter),
            k_tmp: calloc_counted(&mut counter),
            // surround_buffer_x: calloc(&mut counter),
            // surround_buffer_y: calloc(&mut counter),
            // surround_buffer_k: calloc(&mut counter),
            // surround_buffer_len: 0,
            point_cell: calloc_counted(&mut counter),
            point_cell_offset: calloc_counted(&mut counter),
            cell_count: calloc_counted(&mut counter),
            cell_index: calloc_counted(&mut counter),
            relation_table: calloc_counted(&mut counter),
        };
        println!("TOTAL ALLOCATED: {counter} bytes");
        this.x0_tmp.copy_from_slice(&**x0);
        this.y0_tmp.copy_from_slice(&**y0);
        this.x1_tmp.copy_from_slice(&**x1);
        this.y1_tmp.copy_from_slice(&**y1);
        this.k_tmp.copy_from_slice(&**k);
        this.relation_table.copy_from_slice(&*relation_table);
        drop(x0);
        drop(y0);
        drop(x1);
        drop(y1);
        drop(k);

        this.cell_count.iter_mut().for_each(|e| *e = 0.into());
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let new_x = this.x1_tmp[i];
            let new_y = this.y1_tmp[i];
            let cell = compute_cell(new_x, new_y);
            this.point_cell[i] = cell;
            this.point_cell_offset[i] = this.cell_count[cell];
            this.cell_count[cell] += 1.into();
        }
        let mut acc: Pid = 0.into();
        for i in (0..CELLS_PLUS_1_U).map(Into::into) {
            this.cell_index[i] = acc;
            let count = this.cell_count[i];
            acc += count;
        }
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let cell = this.point_cell[i];
            let index = this.cell_index[cell] + this.point_cell_offset[i];
            this.x0[index] = this.x0_tmp[i];
            this.y0[index] = this.y0_tmp[i];
            this.x1[index] = this.x1_tmp[i];
            this.y1[index] = this.y1_tmp[i];
            this.k[index] = this.k_tmp[i];
        }
        this.cell_count.iter_mut().for_each(|e| *e = 0.into());
        this
    }
    fn serialize_gpu(&self, data: &mut Vec<GpuPoint>) {
        data.clear();
        data.extend(
            self.x0
                .iter()
                .zip(self.y0.iter())
                .zip(self.k.iter())
                .map(|((&x, &y), &k)| GpuPoint {
                    x: (x - MIN_X) * (1.0 / (MAX_X - MIN_X)),
                    y: (y - MIN_X) * (1.0 / (MAX_X - MIN_X)),
                    k: k as _,
                    _unused: 0,
                }),
        );
    }
    #[no_panic]
    fn z_to_xy(val: u32) -> (u32, u32) {
        #[allow(clippy::all)]
        const _A: () = assert!(CELLS_X == CELLS_Y);
        use core::arch::x86_64::*;
        const MASK_X: u32 = 0b10101010101010101010101010101010;
        const MASK_Y: u32 = 0b01010101010101010101010101010101;
        unsafe { (_pext_u32(val, MASK_X), _pext_u32(val, MASK_Y)) }
    }

    #[allow(unused_parens)]
    #[inline(never)]
    pub fn update(&mut self) {
        const BLOCK_SIZE: u16 = 4;

        //{{ for cx in 0..CELLS_X {for cy in 0..CELLS_Y {

        // 340, 243
        // 347, 238

        {
            {
                {
                    for z in 0..(CELLS_X as u32 * CELLS_Y as u32) {
                        let (cx, cy) = Self::z_to_xy(z);
                        let (cx, cy) = (cx as u16, cy as u16);
                        // for ccx in (0..CELLS_X).step_by(BLOCK_SIZE as usize) {
                        //     for ccy in (0..CELLS_Y).step_by(BLOCK_SIZE as usize) {
                        //         for cy in ccy..(CELLS_Y.min(ccy + BLOCK_SIZE)) {
                        //             for cx in ccx..(CELLS_X.min(ccx + BLOCK_SIZE)) {
                        //for i in (0..NUM_POINTS_U).map(Into::into) {
                        let cell: Cid = Cid((cx + cy * CELLS_X) as IndexType);
                        //self.fill_surround_buffer(cy, cx);

                        let ranges = self.generate_ranges(cy, cx);

                        for i_self in (usize::from(self.cell_index[cell])
                            ..usize::from(self.cell_index[cell + 1.into()]))
                            .map(Into::into)
                        {
                            let x0 = self.x0[i_self];
                            let y0 = self.y0[i_self];
                            let x1 = self.x1[i_self];
                            let y1 = self.y1[i_self];
                            let k = self.k[i_self];
                            let mut acc_x = 0.0;
                            let mut acc_y = 0.0;

                            // self.k
                            // self.x1
                            // self.y1
                            #[rustfmt::skip]
                            unsafe {
                                use core::arch::x86_64::*;
                                // for range in &ranges {
                                //     _mm_prefetch(self.k.0.get_unchecked(range.start as usize) as *const u8 as *const i8, _MM_HINT_T0);
                                //     _mm_prefetch(self.x1.0.get_unchecked(range.start as usize) as *const f32 as *const i8, _MM_HINT_T0);
                                //     _mm_prefetch(self.y1.0.get_unchecked(range.start as usize) as *const f32 as *const i8, _MM_HINT_T0);
                                // }
                            };

                            // about 42 % faster
                            #[cfg(not(feature = "nosimd"))]
                            unsafe {
                                //self.iterate_neighbours(x1, y1, k, &mut acc_x, &mut acc_y);
                                self.iterate_ranges(
                                    ranges.clone(),
                                    x1,
                                    y1,
                                    k,
                                    &mut acc_x,
                                    &mut acc_y,
                                );
                            }

                            // about 52 % of execution time is spent here
                            #[cfg(feature = "nosimd")]
                            {
                                acc_x = 0.0;
                                acc_y = 0.0;
                                let local_k_arr: [f32; NUM_KINDS] =
                                    *unsafe { self.relation_table.get_unchecked(k as usize) };
                                for ((&x_other, &y_other), &k_other) in self
                                    .surround_buffer_x
                                    .slice()
                                    .iter()
                                    .zip(self.surround_buffer_y.slice().iter())
                                    .zip(self.surround_buffer_k.slice().iter())
                                {
                                    // https://www.desmos.com/calculator/yos22615bv

                                    let dx = x1 - x_other;
                                    let dy = y1 - y_other;
                                    let dot = dx * dx + dy * dy;
                                    let r = dot.sqrt();

                                    let x = r * (1.0 / POINT_MAX_RADIUS);
                                    let k: f32 =
                                        *unsafe { local_k_arr.get_unchecked(k_other as usize) };
                                    let c = 4.0 * (1.0 - x / BETA);
                                    //let c = 10.0 * (1.0 / x - 1.0 / (BETA));

                                    let f = ((k / (1.0 - BETA)) * (x - BETA).min(1.0 - x).max(0.0))
                                        .max(c);
                                    let dir_x = dx / r;
                                    let dir_y = dy / r;
                                    if dot != 0.0 {
                                        acc_x += dir_x * f;
                                        acc_y += dir_y * f;
                                        debug_assert_float!(acc_x);
                                        debug_assert_float!(acc_y);
                                    }
                                }
                            }
                            let (new_x, new_y) =
                                Self::calc_new_position(acc_x, acc_y, x1, x0, y1, y0);

                            self.store_temporary_results(i_self, new_x, new_y);
                        }
                    }
                }
            }
        }

        // The last passes are very serial, but is fast enough
        // that it basically does not matter

        self.prefix_sum();

        self.final_insert();

        swap(&mut self.x1, &mut self.x0_tmp);
        swap(&mut self.y1, &mut self.y0_tmp);
        swap(&mut self.k, &mut self.k_tmp);
    }

    #[inline(never)]
    fn final_insert(&mut self) {
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let cell = self.point_cell[i];
            let index = self.cell_index[cell] + self.point_cell_offset[i];
            // self.x0[index] = self.x0_tmp[i];
            // self.y0[index] = self.y0_tmp[i];
            // self.x1[index] = self.x1_tmp[i];
            // self.y1[index] = self.y1_tmp[i];

            self.x0[index] = self.x1[i];
            self.y0[index] = self.y1[i];
            self.x0_tmp[index] = self.x1_tmp[i];
            self.y0_tmp[index] = self.y1_tmp[i];
            self.k_tmp[index] = self.k[i];
        }
    }

    #[inline(never)]
    fn prefix_sum(&mut self) {
        // let mut arr = [0_u32; 50];

        // for i in 0..CELLS_PLUS_1_U {
        //     let cc = |i: usize| self.cell_count.get(i).map(|i| i.0 as usize).unwrap_or(0);
        //     let tracked_value = cc(i - 1 + CELLS_X as usize)
        //         + cc(i + CELLS_X as usize)
        //         + cc(i + 1 + CELLS_X as usize)
        //         + cc(i - 1)
        //         + cc(i)
        //         + cc(i + 1)
        //         + cc(i - 1 - CELLS_X as usize)
        //         + cc(i - CELLS_X as usize)
        //         + cc(i + 1 - CELLS_X as usize);
        //     *if let Some(a) = arr.get_mut(usize::from(tracked_value)) {
        //         a
        //     } else {
        //         arr.last_mut().unwrap()
        //     } += 1;
        // }

        // let max_val = arr.iter().copied().max().unwrap();
        // let mut s = String::new();
        // for (i, a) in arr.into_iter().enumerate() {
        //     write!(&mut s, "{i}: {a}").unwrap();
        //     let dots = 200.0 * (a as f64 / max_val as f64);
        //     for _ in 0..(dots as usize) {
        //         write!(&mut s, ".").unwrap();
        //     }
        //     write!(&mut s, "\n").unwrap();
        // }
        // println!("\n{s}\n");

        let mut acc: Pid = 0.into();
        for i in (0..CELLS_PLUS_1_U).map(Cid::from) {
            self.cell_index[i] = acc;
            let count = replace(&mut self.cell_count[i], 0.into());
            acc += count;
        }
    }

    #[inline(never)]
    fn store_temporary_results(&mut self, i_self: Pid, new_x: f32, new_y: f32) {
        self.x1_tmp[i_self] = new_x;
        self.y1_tmp[i_self] = new_y;

        // self.x0_tmp[i] = x1;
        // self.y0_tmp[i] = y1;

        let new_cell = compute_cell(new_x, new_y);
        self.point_cell[i_self] = new_cell;
        self.point_cell_offset[i_self] = self.cell_count[new_cell];
        self.cell_count[new_cell] += 1.into();
    }

    #[inline(never)]
    unsafe fn iterate_ranges(
        &mut self,
        ranges: [Range<u32>; 3],
        x1: f32,
        y1: f32,
        k: u8,
        acc_x: &mut f32,
        acc_y: &mut f32,
    ) {
        use simd::*;
        use std::arch::x86_64::*;
        let mut acc_x_s = zero();
        let mut acc_y_s = zero();

        let x1_self = set1(x1);
        let y1_self = set1(y1);
        // 256 = 8xf32 => 8 iterations
        // step_by rounds down
        let k_base: __m256 = _mm256_load_ps(self.relation_table.as_ptr().add(k as usize).cast());

        // TODO: only do first N iterations.

        for range in ranges {
            if range.is_empty() {
                continue;
            }

            for i in range.step_by(8) {
                // 3 loads per iteration

                // load ks, zero extend, and index into base
                //_mm256_castps128_ps256
                let k_factor: __m256 = _mm256_permutevar8x32_ps(
                    k_base,
                    _mm256_cvtepu8_epi32(_mm_loadu_si128(
                        self.k.as_mut_ptr().add(i as _) as *const __m128i
                    )),
                );
                let x_other = _mm256_loadu_ps(self.x1.as_ptr().add(i as _));
                let y_other = _mm256_loadu_ps(self.y1.as_ptr().add(i as _));

                let dx = fma!(x1_self - x_other);
                let dy = fma!(y1_self - y_other);
                //let dot = fmadd(dx, dx, mul(dy, dy));
                let dot = fma!(dx * dx + fma!(dy * dy));
                let recip_sqrt = _mm256_rsqrt_ps(dot); // 1.0 / sqrt(d * d)

                let r = fma!(recip_sqrt * dot); //sqrt(dot);
                                                //let x = mul(r, set1(1.0 / POINT_MAX_RADIUS));

                let x = fma!(r * _mm256_set1_ps(1.0 / POINT_MAX_RADIUS));

                // max(x, y) = 0.5 * (x + y + abs(x - y))
                // min(x, y) = 0.5 * (x + y - abs(x - y))
                // clamp(x, y (min), z (max)) = max(min(x, z), y)
                // = max(0.5 * (x + z - abs(x - z)), y)

                // > 0.0 will also test for Nan
                //let mask_ok = _mm256_cmp_ps(dot, set1(0.0), _CMP_GT_OQ);
                //let mask_ok = _mm256_cmp_ps(dot, set1(0.0), _CMP_NEQ_OQ);
                let mask_ok = _mm256_cmp_ps(dot, _mm256_setzero_ps(), _CMP_NEQ_OQ);

                let f = fma!(
                    recip_sqrt
                        * max(
                            fma!(
                                k_factor * maxzero(min(fma!(x - set1(BETA)), fma!(-x + set1(1.0))))
                            ),
                            fma!(-r * (set1((4.0 / BETA) * (1.0 / POINT_MAX_RADIUS))) + set1(4.0)),
                        )
                );

                // f / r
                let f_masked = fma!(mask_ok & f);

                acc_x_s = fma!(dx * f_masked + acc_x_s);
                acc_y_s = fma!(dy * f_masked + acc_y_s);
            }
        }
        *acc_x = hadd(acc_x_s);
        *acc_y = hadd(acc_y_s);
    }
    /*
    #[inline(never)]
    unsafe fn iterate_neighbours(
        &mut self,
        x1: f32,
        y1: f32,
        k: u8,
        acc_x: &mut f32,
        acc_y: &mut f32,
    ) {
        use simd::*;
        use std::arch::x86_64::*;
        let mut acc_x_s = zero();
        let mut acc_y_s = zero();

        let x1_self = set1(x1);
        let y1_self = set1(y1);
        // 256 = 8xf32 => 8 iterations
        // step_by rounds down
        let k_base: __m256 = _mm256_load_ps(self.relation_table.as_ptr().add(k as usize).cast());

        // TODO: only do first N iterations.

        for i in (0..self.surround_buffer_len).step_by(8) {
            // load ks, zero extend, and index into base
            let k_factor: __m256 = _mm256_permutevar8x32_ps(
                k_base,
                _mm256_cvtepu8_epi32(_mm256_castsi256_si128({
                    _mm256_loadu_si256(
                        self.surround_buffer_k.inner.as_mut_ptr().add(i) as *const __m256i
                    )
                })),
            );
            let x_other = load(self.surround_buffer_x.as_ptr().add(i));
            let y_other = load(self.surround_buffer_y.as_ptr().add(i));

            let dx = fma!(x1_self - x_other);
            let dy = fma!(y1_self - y_other);
            //let dot = fmadd(dx, dx, mul(dy, dy));
            let dot = fma!(dx * dx + fma!(dy * dy));
            let recip_sqrt = _mm256_rsqrt_ps(dot); // 1.0 / sqrt(d * d)

            let r = fma!(recip_sqrt * dot); //sqrt(dot);
                                            //let x = mul(r, set1(1.0 / POINT_MAX_RADIUS));
            let x = fma!(r * set1(1.0 / POINT_MAX_RADIUS));

            // max(x, y) = 0.5 * (x + y + abs(x - y))
            // min(x, y) = 0.5 * (x + y - abs(x - y))
            // clamp(x, y (min), z (max)) = max(min(x, z), y)
            // = max(0.5 * (x + z - abs(x - z)), y)

            // > 0.0 will also test for Nan
            //let mask_ok = _mm256_cmp_ps(dot, set1(0.0), _CMP_GT_OQ);
            //let mask_ok = _mm256_cmp_ps(dot, set1(0.0), _CMP_NEQ_OQ);
            let mask_ok = _mm256_cmp_ps(dot, zero(), _CMP_NEQ_OQ);

            let f = fma!(
                recip_sqrt
                    * max(
                        fma!(k_factor * maxzero(min(fma!(x - set1(BETA)), fma!(-x + set1(1.0))))),
                        fma!(-r * (set1((4.0 / BETA) * (1.0 / POINT_MAX_RADIUS))) + set1(4.0)),
                    )
            );

            // f / r
            let f_masked = fma!(mask_ok & f);

            acc_x_s = fma!(dx * f_masked + acc_x_s);
            acc_y_s = fma!(dy * f_masked + acc_y_s);
        }
        *acc_x = hadd(acc_x_s);
        *acc_y = hadd(acc_y_s);
    }*/

    #[inline(never)]
    fn generate_ranges(&self, cy: u16, cx: u16) -> [Range<u32>; 3] {
        [-1, 0, 1].map(|dy| {
            let cell_y = cy as i32 + dy;
            if cell_y == -1 || cell_y >= CELLS_Y as i32 {
                0..0
            } else {
                let cell_x_min = cx as i32 - 1;
                let cell_x_max = cx as i32 + 1;
                let cell_x_min_cropped = cell_x_min.max(0_i32);
                let cell_x_max_cropped = cell_x_max.min(CELLS_X as i32 - 1);

                let cell_min_cropped: Cid =
                    ((cell_x_min_cropped + cell_y * CELLS_X as i32) as usize).into();
                let cell_max_cropped: Cid =
                    ((cell_x_max_cropped + cell_y * CELLS_X as i32) as usize).into();

                u32::from(self.cell_index[cell_min_cropped])
                    ..u32::from(self.cell_index[cell_max_cropped + 1.into()])
            }
        })
    }

    /*#[inline(never)]
    fn fill_surround_buffer(&mut self, cy: u16, cx: u16) {
        self.surround_buffer_len = 0;
        for dy in -1..=1 {
            let cell_y = cy as i32 + dy;
            if cell_y == -1 {
                continue;
            } else if cell_y == CELLS_Y as i32 {
                continue;
            }

            let cell_x_min = cx as i32 - 1;
            let cell_x_max = cx as i32 + 1;
            let cell_x_min_cropped = cell_x_min.max(0_i32);
            let cell_x_max_cropped = cell_x_max.min(CELLS_X as i32 - 1);

            let cell_min_cropped: Cid =
                ((cell_x_min_cropped + cell_y * CELLS_X as i32) as usize).into();
            let cell_max_cropped: Cid =
                ((cell_x_max_cropped + cell_y * CELLS_X as i32) as usize).into();

            let cropped_range = usize::from(self.cell_index[cell_min_cropped])
                ..usize::from(self.cell_index[cell_max_cropped + 1.into()]);

            // let range_middle_leftover8 = range_middle.len() % 8;
            // let range_middle_leftover32 = range_middle.len() % 32;

            #[cfg(not(feature = "nosimd"))]
            unsafe {
                use core::arch::x86_64::*;
                let mut x_write_ptr = self.surround_buffer_x.end_ptr_mut(self.surround_buffer_len);
                let mut y_write_ptr = self.surround_buffer_y.end_ptr_mut(self.surround_buffer_len);
                let mut k_write_ptr = self.surround_buffer_k.end_ptr_mut(self.surround_buffer_len);

                core::ptr::copy_nonoverlapping(
                    self.x1.as_ptr().add(cropped_range.start),
                    x_write_ptr,
                    cropped_range.len(),
                );
                core::ptr::copy_nonoverlapping(
                    self.y1.as_ptr().add(cropped_range.start),
                    y_write_ptr,
                    cropped_range.len(),
                );

                core::ptr::copy_nonoverlapping(
                    self.k.as_ptr().add(cropped_range.start),
                    k_write_ptr,
                    cropped_range.len(),
                );

                // for i in range_middle.clone().map(Into::into).step_by(8) {
                //     let x_read_ptr = self.x1.as_ptr().add(i);
                //     let y_read_ptr = self.y1.as_ptr().add(i);
                //     let k_read_ptr = self.k.as_ptr().add(i);
                //     // TODO: rep movsb
                //     _mm256_storeu_ps(x_write_ptr, _mm256_loadu_ps(x_read_ptr));
                //     _mm256_storeu_ps(
                //         y_write_ptr,
                //         _mm256_loadu_ps(y_read_ptr),
                //         //_mm256_add_ps(offset_y, _mm256_loadu_ps(y_read_ptr)),
                //     );
                //     (k_write_ptr as *mut u64)
                //         .write_unaligned((k_read_ptr as *mut u64).read_unaligned());
                //     x_write_ptr = x_write_ptr.add(8);
                //     y_write_ptr = y_write_ptr.add(8);
                //     k_write_ptr = k_write_ptr.add(8);
                // }
                self.surround_buffer_len += cropped_range.len();
            }
            #[cfg(feature = "nosimd")]
            for i in range_middle.map(Into::into) {
                unsafe {
                    self.surround_buffer_x.push(self.x1[i]);
                    self.surround_buffer_y.push(self.y1[i] + offset_y);
                    self.surround_buffer_k.push(self.k[i]);
                }
            }
        }
        #[cfg(not(feature = "nosimd"))]
        unsafe {
            use core::arch::x86_64::*;
            let zero = _mm256_setzero_ps();
            _mm256_storeu_ps(
                self.surround_buffer_x.end_ptr_mut(self.surround_buffer_len),
                zero,
            );
            _mm256_storeu_ps(
                self.surround_buffer_y.end_ptr_mut(self.surround_buffer_len),
                zero,
            );
        }
    }*/
    #[inline(never)]
    fn calc_new_position(
        mut acc_x: f32,
        mut acc_y: f32,
        x1: f32,
        x0: f32,
        y1: f32,
        y0: f32,
    ) -> (f32, f32) {
        debug_assert_float!(acc_x);
        debug_assert_float!(acc_y);
        debug_assert_float!(acc_x);
        debug_assert_float!(acc_y);
        acc_x *= MIN_WIDTH;
        acc_y *= MIN_WIDTH;

        /*
        n in [-1, 0, 1]
        dx = x1 - x0 + n * MIN_WIDTH;
        dy = y1 - y0 + n * MIN_WIDTH;

        want closest value to zero
        => ignore sign, then add sign back
        */

        let vx = x1 - x0;
        let vy = y1 - y0;
        let v2 = f32::mul_add(vx, vx, vy * vy);
        let v = v2.sqrt();
        let v_inv = 1.0 / v;
        // let vx_norm = vx * v_inv;
        // let vy_norm = vy * v_inv;
        // let friction_force = v2 * (50.0 / (DT * DT));
        // let friction_force = v2 * (500.0 / (DT * DT));
        //let friction_force = v2 * (500.0 / (DT * DT));
        // acc_x *= 0.001;
        // acc_y *= 0.001;
        //let friction_force = v * v * 31250000.0;

        let friction_force: f32;
        {
            const H: f32 = 0.002;
            const C: f32 = 0.999;
            let x = v.min((2.0 * H) / C);

            let target_v = (C - (C * C * (1.0 / (4.0 * H)) * x)) * x;
            friction_force = v - target_v;
        }
        let factor = (friction_force * (1.0 / (DT * DT))) * v_inv;
        if !(vx.is_nan() || vy.is_nan() || factor.is_nan()) {
            acc_x -= factor * vx;
            acc_y -= factor * vy;
        }
        // assert!(!acc_x.is_nan());
        // assert!(!acc_y.is_nan());
        // assert!(!x1.is_nan());
        // assert!(!y1.is_nan());
        // assert!(!vx.is_nan());
        // assert!(!vy.is_nan());

        let new_x = reflect_sim_x(f32::mul_add(acc_x, DT * DT, x1) + vx);
        //.rem_euclid(1.0);
        let new_y = reflect_sim_y(f32::mul_add(acc_y, DT * DT, y1) + vy);
        //.rem_euclid(1.0);
        // if !(debug_check_valid_pos(x1)
        //     && debug_check_valid_pos(y1)
        //     && debug_check_valid_pos(new_x)
        //     && debug_check_valid_pos(new_y))
        // {
        //     panic!("ERR: x1: {x1}, y1: {y1}, vx: {vx}, vy: {vy}, acc_x: {acc_x}, acc_y: {acc_y} new_x: {new_x}, new_y: {new_y}");
        // }
        (new_x, new_y)
    }
}

macro_rules! printlnc {
    ($aa:expr, $bb:expr) => {
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(&format!($aa, $bb).into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!($aa, $bb);
        }
    };
}
macro_rules! dbgc {
    ($aa:expr) => {{
        match $aa {
            tmp => {
                eprintln!(
                    "[{}:{}] {} = {:#?}",
                    file!(),
                    line!(),
                    stringify!($aa),
                    &tmp
                );
                tmp
            },
        }
    }};
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let (event_loop, window) = window_setup();

    let mut state = State::new(window).await;
    event_loop.run(move |os_event, _, control_flow| match os_event {
        Event::WindowEvent {
            event: window_event,
            window_id,
        } if window_id == state.window.id() => match window_event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => state.resize(*new_inner_size),
            _ => state.input(&window_event),
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            match state.render() {
                Ok(()) => {},
                Err(SurfaceError::Lost) => state.resize(state.size),
                Err(SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{e:?}"),
            }
        },
        Event::MainEventsCleared => {
            state.window().request_redraw();
        },
        _ => {},
    });
}

fn window_setup() -> (EventLoop<()>, Window) {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("id-thingy")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }
    (event_loop, window)
}
struct State {
    params: Params,
    keystate: KeyState,

    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,

    #[cfg(not(target_arch = "wasm32"))]
    last_print: Instant,
    frame: usize,

    //simulation_state: SimulationState,
    simulation_controller: SimThreadController,

    point_transfer_buffer: Vec<GpuPoint>,
    gpu_point_buffer: wgpu::Buffer,
    vertex_offsets_buffer: wgpu::Buffer,
    vertices: u32,

    params_buffer: wgpu::Buffer,
    //gpu_simulation_state: GpuSimulationState,
}

impl State {
    async fn new(window: Window) -> Self {
        //let simulation_state = SimulationState::new(SimulationInputEuler::new().verlet());
        let mut simulation_controller = SimThreadController::new();

        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        });
        dbgc!(instance
            .enumerate_adapters(wgpu::Backends::all())
            .collect::<Vec<_>>());
        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter: wgpu::Adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .find(|adapter| adapter.is_surface_supported(&surface))
            .unwrap();
        dbgc!(adapter.features());
        dbgc!(adapter.limits());

        let limits = wgpu::Limits::downlevel_defaults(); //downlevel_webgl2_defaults();

        dbgc!(&limits);
        println!(
            "Expected points per cell: {}",
            NUM_POINTS as f32 / CELLS as f32
        );
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: adapter.features(), //wgpu::Features::VERTEX_WRITABLE_STORAGE,
                    limits: adapter.limits(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format: wgpu::TextureFormat = surface_caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(*surface_caps.formats.first().unwrap());
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            //present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: *surface_caps.alpha_modes.first().unwrap(),
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let mut point_transfer_buffer = Vec::new();
        //simulation_state.serialize_gpu(&mut point_transfer_buffer);
        simulation_controller.get_state(&mut point_transfer_buffer);
        let gpu_point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Buffer"),
            contents: bytemuck::cast_slice(&point_transfer_buffer),
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX,
        });

        //let gpu_simulation_state = {
        //    let simulation_state = CpuSimulationState::new();
        //    GpuSimulationState::from_simulation_state(&device, &gpu_point_buffer, &simulation_state)
        //};

        let params = Params::new();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Param Buffer"),
            contents: bytemuck::cast_slice(&bytemuck::cast::<_, [u8; size_of::<Params>()]>(params)),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let tri_size = POINT_MAX_RADIUS * BETA / (MAX_X - MIN_X); //0.003;

        // let vertex_offsets: [f32; 6] = {
        //     [
        //         f32::sqrt(3.0) / 2.0,
        //         -0.5,
        //         0.0,
        //         1.0,
        //         -f32::sqrt(3.0) / 2.0,
        //         -0.5,
        //     ]
        //     .map(|x| x * tri_size)
        // };
        let qq: i32 = 32;
        let vertex_offsets: Vec<_> = (0_i32..qq)
            .map(|i| ((i as f32) / ((qq - 1_i32) as f32)) * core::f32::consts::PI * 2.0)
            .flat_map(|i| [-i.sin(), i.cos()])
            .map(|i| i * tri_size)
            .collect();
        let vertices = u32::try_from(vertex_offsets.len().checked_div(2).unwrap()).unwrap();
        let vertex_offsets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&vertex_offsets),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let bind_group_layout: wgpu::BindGroupLayout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout"),
                entries: &[
                    //wgpu::BindGroupLayoutEntry {
                    //    binding: 0,
                    //    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    //    ty: wgpu::BindingType::Buffer {
                    //        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    //        has_dynamic_offset: false,
                    //        min_binding_size: Default::default(),
                    //    },
                    //    count: None,
                    //},
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &bind_group_layout,
            entries: &[
                //wgpu::BindGroupEntry {
                //    binding: 0,
                //    resource: wgpu::BindingResource::Buffer(
                //        gpu_point_buffer.as_entire_buffer_binding(),
                //    ),
                //},
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(
                        params_buffer.as_entire_buffer_binding(),
                    ),
                },
            ],
        });

        let shader_source: String = std::fs::read_to_string("src/shader.wgsl")
            .unwrap()
            .replace("{NUM_POINTS}", &format!("{NUM_POINTS}"))
            .replace("{NUM_KINDS}", &format!("{NUM_KINDS}"))
            .replace("{WIDTH_X}", &format!("{WIDTH_X:.10}"))
            .replace("{WIDTH_Y}", &format!("{WIDTH_Y:.10}"));

        let render_pipeline;
        {
            let fragment_vertex_shader: wgpu::ShaderModule =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Fragment and vertex shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });
            let pipeline_layout: wgpu::PipelineLayout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
            let layout = Some(&pipeline_layout);
            let multisample: wgpu::MultisampleState = wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            };

            render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout,
                    vertex: wgpu::VertexState {
                        module: &fragment_vertex_shader,
                        entry_point: "vs_main",
                        buffers: &[
                            wgpu::VertexBufferLayout {
                                array_stride: size_of::<GpuPoint>() as u64,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Uint32, 2 => Uint32],
                            },
                            wgpu::VertexBufferLayout {
                                array_stride: size_of::<f32>() as u64 * 2,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &wgpu::vertex_attr_array![3 => Float32x2],
                            },
                        ],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fragment_vertex_shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: config.format,
                            blend: None,
                            //blend: Some(wgpu::BlendState { 
                            //    color: wgpu::BlendComponent { 
                            //        src_factor: wgpu::BlendFactor::SrcAlpha, 
                            //        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha, 
                            //        operation: wgpu::BlendOperation::Add, 
                            //    }, 
                            //    alpha: wgpu::BlendComponent::default() 
                            //}),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::LineStrip, //TriangleList
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,//Some(wgpu::Face::Back),
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample,
                    multiview: None,
                });
        }

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            bind_group,
            #[cfg(not(target_arch = "wasm32"))]
            last_print: Instant::now(),
            frame: 0,
            simulation_controller,
            point_transfer_buffer,
            gpu_point_buffer,
            params,
            params_buffer,
            keystate: KeyState::new(),
            //gpu_simulation_state,
            vertex_offsets_buffer,
            vertices,
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        {
            let control_dt = 0.03;

            let k = &self.keystate;

            let lr = i32::from(k.right) as f32 - i32::from(k.left) as f32;
            let ud = i32::from(k.up) as f32 - i32::from(k.down) as f32;
            let io = i32::from(k.zoom_out) as f32 - i32::from(k.zoom_in) as f32;
            self.params.offset_x += lr * control_dt / self.params.zoom_x;
            self.params.offset_y += ud * control_dt / self.params.zoom_x;

            self.params.zoom_x += io * control_dt * self.params.zoom_x;
            //self.params.zoom_x = self.params.zoom_x.max(1.0);
            self.params.zoom_y =
                self.params.zoom_x * self.size.width as f32 / (self.size.height as f32);
        }

        self.frame += 1;
        let max_frame = 240;
        #[cfg(not(target_arch = "wasm32"))]
        if self.frame == max_frame {
            self.frame = 0;
            let dt = replace(&mut self.last_print, Instant::now()).elapsed();
            let fps = max_frame as f64 / dt.as_secs_f64();
            printlnc!("fps: {fps:?}, dt: {:50?}", dt);
        }

        let output: wgpu::SurfaceTexture = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        if !USE_GPU_COMPUTE {
            self.queue.write_buffer(
                &self.gpu_point_buffer,
                0,
                bytemuck::cast_slice(&self.point_transfer_buffer),
            );
            self.simulation_controller
                .try_get_state(&mut self.point_transfer_buffer);
        }
        self.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&bytemuck::cast::<_, [u8; size_of::<Params>()]>(self.params)),
        );

        let mut encoder: wgpu::CommandEncoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder"),
                });
        //if USE_GPU_COMPUTE {
        //    let mut compute_pass =
        //        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        //    let compute_pass = &mut compute_pass;
        //    self.gpu_simulation_state.update(compute_pass);
        //    self.gpu_simulation_state.serialize_gpu(compute_pass);
        //}

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            //render_pass.set_vertex_buffer(0, &self.gpu_point_buffer.slice(..));
            //render_pass.draw(0..(3 * NUM_POINTS * 9), 0..1);
            //render_pass.draw(0..(3 * NUM_POINTS), 0..1);
            render_pass.set_vertex_buffer(0, self.gpu_point_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.vertex_offsets_buffer.slice(..));
            render_pass.draw(0..self.vertices, 0..NUM_POINTS);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn input(&mut self, event: &WindowEvent<'_>) {
        if let &WindowEvent::KeyboardInput {
            input:
                winit::event::KeyboardInput {
                    virtual_keycode: Some(keycode),
                    state,
                    ..
                },
            ..
        } = event
        {
            use VirtualKeyCode::*;
            let pressed = state == ElementState::Pressed;

            match keycode {
                W => self.keystate.up = pressed,
                S => self.keystate.down = pressed,
                A => self.keystate.left = pressed,
                D => self.keystate.right = pressed,
                Space => self.keystate.zoom_out = pressed,
                LShift => self.keystate.zoom_in = pressed,
                _ => (),
            };
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}

mod simd {
    #![allow(dead_code)]
    use core::arch::x86_64::*;
    use no_panic::no_panic;
    #[inline(always)]
    #[no_panic]
    pub unsafe fn load(a: *const f32) -> __m256 {
        _mm256_load_ps(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn loadu(a: *const f32) -> __m256 {
        _mm256_loadu_ps(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn loadi(a: *const __m256i) -> __m256i {
        _mm256_load_si256(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn loadiu(a: *const __m256i) -> __m256i {
        _mm256_loadu_si256(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn add(a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn andi(a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn and(a: __m256, b: __m256) -> __m256 {
        _mm256_and_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn andn(a: __m256, b: __m256) -> __m256 {
        _mm256_andnot_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn sub(a: __m256, b: __m256) -> __m256 {
        _mm256_sub_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn fmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn fmsub(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn fnmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fnmadd_ps(a, b, c)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn fnmsub(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fnmsub_ps(a, b, c)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn gather<const SCALE: i32>(p: *const f32, indexes: __m256i) -> __m256 {
        _mm256_i32gather_ps::<SCALE>(p, indexes)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn sqrt(a: __m256) -> __m256 {
        _mm256_sqrt_ps(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn div(a: __m256, b: __m256) -> __m256 {
        _mm256_div_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn rsqrt_fast(a: __m256) -> __m256 {
        _mm256_rsqrt_ps(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn zero() -> __m256 {
        _mm256_setzero_ps()
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn set1(a: f32) -> __m256 {
        _mm256_set1_ps(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn splati(a: i32) -> __m256i {
        _mm256_set1_epi32(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn max(a: __m256, b: __m256) -> __m256 {
        _mm256_max_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn maxzero(a: __m256) -> __m256 {
        //_mm256_max_ps(a, zero())
        _mm256_blendv_ps(a, zero(), a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn min(a: __m256, b: __m256) -> __m256 {
        _mm256_min_ps(a, b)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn slli<const IMM: i32>(a: __m256i) -> __m256i {
        _mm256_slli_epi32::<IMM>(a)
    }
    #[inline(always)]
    #[no_panic]
    pub unsafe fn hadd(a: __m256) -> f32 {
        let mut b: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(b.as_mut_ptr(), a);
        ((b[0] + b[1]) + (b[2] + b[3])) + ((b[4] + b[5]) + (b[6] + b[7]))
    }
    // #[inline(always)]
    // pub unsafe fn abs(a: __m256) -> __m256 {
    //     let sign_bit = 0b1000_0000__0000_0000__0000_0000__0000_0000;
    //     let mask = _mm256_set1_epi32(!sign_bit);
    //     _mm256_castsi256_ps()
    //     todo!()

    // }

    pub fn as_rchunks<T, const N: usize>(this: &[T]) -> (&[T], &[[T; N]]) {
        unsafe fn as_chunks_unchecked<T, const N: usize>(s: &[T]) -> &[[T; N]] {
            let new_len = s.len() / N;
            unsafe { core::slice::from_raw_parts(s.as_ptr().cast(), new_len) }
        }
        assert!(N != 0, "chunk size must be non-zero");
        let len = this.len() / N;
        let (remainder, multiple_of_n) = this.split_at(this.len() - len * N);
        // SAFETY: We already panicked for zero, and ensured by construction
        // that the length of the subslice is a multiple of N.
        let array_slice = unsafe { as_chunks_unchecked(multiple_of_n) };
        (remainder, array_slice)
    }
}
