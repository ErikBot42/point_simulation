#![feature(portable_simd)]

use std::array::from_fn;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::mem::{replace, size_of, swap};
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::time::Duration;
use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use prng::Prng;

mod prng;
mod vertex;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Instant, SystemTime};

const CELLS_X: u16 = 64;
const CELLS_Y: u16 = 64;
const CELLS_MAX_DIM: u16 = if CELLS_X > CELLS_Y { CELLS_X } else { CELLS_Y };
const CELLS: u16 = CELLS_X * CELLS_Y;
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
#[cfg(not(feature = "simd"))]
const K_FAC: f32 = 2.0 * 0.2;
#[cfg(feature = "simd")]
const K_FAC: f32 = 2.0 * 0.2 * (1.0 / (1.0 - BETA));
const BETA: f32 = 0.3; //0.3;
const DT: f32 = 0.008; // TODO: define max speed/max dt from cell size.
const NUM_POINTS_U: usize = NUM_POINTS as usize;
const CELLS_PLUS_1_U: usize = CELLS as usize + 1;
const NUM_POINTS: u32 = 8192; //2048; //16384;
const NUM_KINDS: usize = 8;

fn debug_check_valid_pos(f: f32) -> bool {
    #[cfg(debug_assertions)]
    let r = f >= MIN_X && f <= MAX_X;
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
fn scale_simulation_x(f: f32) -> f32 {
    MIN_X + (MAX_X - MIN_X) * f
}
fn scale_simulation_y(f: f32) -> f32 {
    MIN_Y + (MAX_Y - MIN_Y) * f
}
fn inv_scale_simulation_x(f: f32) -> f32 {
    (f - MIN_X) / (MAX_X - MIN_X)
}
fn inv_scale_simulation_y(f: f32) -> f32 {
    (f - MIN_Y) / (MAX_Y - MIN_Y)
}
fn mod_simulation_x(f: f32) -> f32 {
    (f - MIN_X).rem_euclid(MAX_X - MIN_X) + MIN_X
}
fn mod_simulation_y(f: f32) -> f32 {
    (f - MIN_Y).rem_euclid(MAX_Y - MIN_Y) + MIN_Y
}

fn compute_cell(x: f32, y: f32) -> Cid {
    let x = inv_scale_simulation_x(x);
    let y = inv_scale_simulation_y(y);
    // x in 0_f32..1_f32
    // y in 0_f32..1_f32

    let cell_x = ((x * CELLS_X as f32) as u16).min(CELLS_X - 1);
    let cell_y = ((y * CELLS_Y as f32) as u16).min(CELLS_Y - 1);
    let cell = cell_x + cell_y * CELLS_X;
    debug_assert!(
        cell < CELLS,
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
    fn new() -> Self {
        Self::from_seed(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
        )
    }
    fn from_seed(seed: u64) -> SimulationInputEuler {
        let mut rng = Prng(seed);
        let xy_range = 0.0..1.0;
        let v_range = -0.1..0.1; //-1.0..1.0;
        fn iter_alloc<T: std::fmt::Debug, const SIZE: usize, F: FnMut() -> T>(
            mut f: F,
        ) -> Box<[T; SIZE]> {
            (0..SIZE)
                .map(|_| f())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        }

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
            *x1 = (x0 + *x1 * DT).rem_euclid(1.0);
            *y1 = (y0 + *y1 * DT).rem_euclid(1.0);
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

macro_rules! index_wrap {
    ($name:ident, $index:ty, $doc:expr) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(Copy, Clone, Default, Debug)]
        struct $name($index);
        unsafe impl bytemuck::Zeroable for $name {}
        unsafe impl bytemuck::Pod for $name {}
        impl From<$name> for usize {
            fn from(value: $name) -> usize {
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
    };
}
index_wrap!(Pid, u16, "Point index");
index_wrap!(Cid, u16, "Cell index");
/// Packed "float" and 4 bit value, exploiting unused bits in a float to save some memory.
#[repr(transparent)]
#[derive(Copy, Clone, Default, Debug)]
struct Packed(u32);
impl Packed {
    fn pack(float: f32, val: u8) -> u32 {
        // float needs to be in the range 64.0_f32..128.0_f32
        // 0b10111101000000000000000000000000
        let mask: u32 = 0b00111100000000000000000000000000;
        let shift = mask.trailing_zeros();
        float.to_bits() | ((val as u32) << shift)
    }
    fn unpack(self) -> (f32, u8) {
        // float needs to be in the range 64.0_f32..128.0_f32
        // 0b10111101000000000000000000000000
        let packed = self.0;
        let mask: u32 = 0b00111100000000000000000000000000;
        let shift = mask.trailing_zeros();
        let float = f32::from_bits(packed & (!mask));
        let val = ((packed & mask) >> shift) as u8;
        (float, val)
    }
}
#[derive(Debug)]
struct Tb<I, T, const SIZE: usize>(Box<[T; SIZE]>, PhantomData<I>);
impl<I, T, const SIZE: usize> From<Box<[T; SIZE]>> for Tb<I, T, SIZE> {
    fn from(value: Box<[T; SIZE]>) -> Tb<I, T, SIZE> {
        Tb(value, PhantomData)
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
    recv: Receiver<Vec<GpuPoint>>,
    counter: Arc<AtomicUsize>,
}
impl SimThreadController {
    fn get_state(&mut self, v: &mut Vec<GpuPoint>) {
        *v = self.recv.recv().unwrap();
    }
    fn try_get_state(&mut self, v: &mut Vec<GpuPoint>) {
        let _ = self.recv.try_recv().map(|r| {
            self.counter.fetch_sub(1, Relaxed);
            *v = r
        });
    }
    fn new() -> Self {
        let (send, recv) = channel();
        let counter = Arc::new(AtomicUsize::new(0));
        let their_counter = counter.clone();
        thread::spawn(|| {
            let send = send;
            let counter = their_counter;
            let mut simulation_state = SimulationState::new();
            let mut delta = Instant::now();
            let mut updates = 0;
            loop {
                simulation_state.update();
                updates += 1;
                let mut v = Vec::new();
                if counter.load(Relaxed) < 5 {
                    simulation_state.serialize_gpu(&mut v);
                    counter.fetch_add(1, Relaxed);
                    send.send(v).unwrap();
                }
                let elapsed = delta.elapsed();
                if elapsed > Duration::from_secs(1) {
                    let ups = updates as f64 / elapsed.as_secs_f64();
                    println!("ups: {ups}");
                    updates = 0;
                    delta = Instant::now();
                }
            }
        });

        SimThreadController { recv, counter }
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
    len: usize,
}

impl<T, const SIZE: usize> From<Box<[T; SIZE]>> for UnsafeVec<T, SIZE> {
    fn from(inner: Box<[T; SIZE]>) -> Self {
        UnsafeVec { inner, len: 0 }
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

impl<T, const SIZE: usize> UnsafeVec<T, SIZE> {
    #[inline(always)]
    fn clear(&mut self) {
        self.len = 0;
    }
    #[inline(always)]
    unsafe fn push(&mut self, element: T) {
        *self.inner.get_unchecked_mut(self.len) = element;
        self.len += 1;
    }
    #[inline(always)]
    fn slice(&self) -> &[T] {
        &unsafe { self.inner.get_unchecked(0..self.len) }
    }
}

fn calloc<T: Default + Debug + bytemuck::Zeroable + bytemuck::Pod, const SIZE: usize, O>() -> O
where
    O: From<Box<[T; SIZE]>>,
{
    //let b: Box<[T; SIZE]> = (0..SIZE)
    //    .map(|_| T::default())
    //    .collect::<Vec<T>>()
    //    .try_into()
    //    .unwrap();
    let b = aligned_alloc();
    let alignment = 1 << (b.as_ptr() as usize).trailing_zeros();
    println!(
        "ALIGNMENT: {}, LEN: {} {}",
        alignment,
        b.len(),
        if alignment >= 64 { "(OK)" } else { "" }
    );
    O::from(b)
}

fn aligned_alloc<T: Default + bytemuck::Zeroable + bytemuck::Pod, const SIZE: usize>(
) -> Box<[T; SIZE]> {
    #[repr(align(64), C)]
    struct AlignedBytes([u8; 64]);

    let mut v = ManuallyDrop::new(
        (0..((SIZE * size_of::<T>()) / size_of::<AlignedBytes>()))
            .map(|_| AlignedBytes([0; 64]))
            .collect::<Vec<AlignedBytes>>(),
    );
    v.shrink_to_fit();
    unsafe { Box::from_raw(v.as_mut_ptr() as *mut [T; SIZE]) }
}

/// Entire state of the simulation
pub struct SimulationState {
    x0: Pbox<f32>,
    y0: Pbox<f32>,
    x1: Pbox<f32>,
    y1: Pbox<f32>,
    k: Pbox<u8>,

    x0_tmp: Pbox<f32>,
    y0_tmp: Pbox<f32>,
    x1_tmp: Pbox<f32>,
    y1_tmp: Pbox<f32>,
    k_tmp: Pbox<u8>,

    surround_buffer_x: UnsafePointVec<f32>,
    surround_buffer_y: UnsafePointVec<f32>,
    surround_buffer_k: UnsafePointVec<u8>,

    point_cell: Pbox<Cid>,

    point_cell_offset: Pbox<Pid>,

    cell_count: Cbox<Pid>,
    cell_index: Cbox<Pid>,

    relation_table: Box<[[f32; NUM_KINDS]; NUM_KINDS]>,
}
impl SimulationState {
    pub fn new() -> Self {
        Self::from_verlet(SimulationInputEuler::new().verlet())
    }
    pub fn from_seed(seed: u64) -> Self {
        Self::from_verlet(SimulationInputEuler::from_seed(seed).verlet())
    }
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
        let mut this = Self {
            x0: calloc(),
            y0: calloc(),
            x1: calloc(),
            y1: calloc(),
            k: calloc(),
            x0_tmp: calloc(),
            y0_tmp: calloc(),
            x1_tmp: calloc(),
            y1_tmp: calloc(),
            k_tmp: calloc(),
            surround_buffer_x: calloc(),
            surround_buffer_y: calloc(),
            surround_buffer_k: calloc(),
            point_cell: calloc(),
            point_cell_offset: calloc(),
            cell_count: calloc(),
            cell_index: calloc(),
            relation_table: calloc(),
        };
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
    #[inline(never)]
    pub fn update(&mut self) {
        // core physics
        for cx in 0..CELLS_X {
            for cy in 0..CELLS_Y {
                //for i in (0..NUM_POINTS_U).map(Into::into) {
                let cell: Cid = usize::from(cx + cy * CELLS_X).into();
                self.surround_buffer_x.clear();
                self.surround_buffer_y.clear();
                self.surround_buffer_k.clear();
                for dy in -1..=1 {
                    let cell_y = cy as i32 + dy as i32;
                    let (offset_y, cell_y) = if cell_y == -1 {
                        (-WIDTH_Y, CELLS_Y as i32 - 1)
                    } else if cell_y == CELLS_Y as i32 {
                        (WIDTH_Y, 0)
                    } else {
                        (0.0, cell_y)
                    };

                    let cell_x_min = cx as i32 - 1;
                    let cell_x_max = cx as i32 + 1;
                    let cell_x_min_cropped = (cell_x_min as i32 - 1).max(0);
                    let cell_x_max_cropped = (cell_x_max as i32 + 1).min(CELLS_X as i32 - 1);

                    let cell_min_cropped: Cid = ((cell_x_min_cropped + cell_y * CELLS_X as i32) as usize).into();
                    let cell_max_cropped: Cid = ((cell_x_max_cropped + cell_y * CELLS_X as i32) as usize).into();

                    let range_middle = usize::from(self.cell_index[cell_min_cropped])
                        ..usize::from(self.cell_index[cell_max_cropped + 1.into()]);
                    

                    for i in range_middle.map(Into::into) {
                        unsafe {
                            self.surround_buffer_x.push(self.x1[i]);
                            self.surround_buffer_y.push(self.y1[i] + offset_y);
                            self.surround_buffer_k.push(self.k[i]);
                        }
                    }
                    //for dx in -1..=1 {
                    //    let cell_x = cx as i32 + dx as i32;
                    //    let (offset_x, cell_x) = if cell_x == -1 {
                    //        (-WIDTH_X, CELLS_X as i32 - 1)
                    //    } else if cell_x == CELLS_X as i32 {
                    //        (WIDTH_X, 0)
                    //    } else {
                    //        (0.0, cell_x)
                    //    };
                    //    let cell: Cid = ((cell_x + cell_y * CELLS_X as i32) as usize).into();

                    //    let range = usize::from(self.cell_index[cell])
                    //        ..usize::from(self.cell_index[cell + 1.into()]);
                    //    for i in range.map(Into::into) {
                    //        unsafe {
                    //            self.surround_buffer_x.push(self.x1[i] + offset_x);
                    //            self.surround_buffer_y.push(self.y1[i] + offset_y);
                    //            self.surround_buffer_k.push(self.k[i]);
                    //        }
                    //    }
                    //}

                    /*for dx in -1..=1 {
                        let cell_x = cx as i32 + dx as i32;
                        let cell_y = cy as i32 + dy as i32;
                        let (offset_x, cell_x) = if cell_x == -1 {
                            (-WIDTH_X, CELLS_X as i32 - 1)
                        } else if cell_x == CELLS_X as i32 {
                            (WIDTH_X, 0)
                        } else {
                            (0.0, cell_x)
                        };
                        let (offset_y, cell_y) = if cell_y == -1 {
                            (-WIDTH_Y, CELLS_Y as i32 - 1)
                        } else if cell_y == CELLS_Y as i32 {
                            (WIDTH_Y, 0)
                        } else {
                            (0.0, cell_y)
                        };
                        let cell: Cid = ((cell_x + cell_y * CELLS_X as i32) as usize).into();

                        let range = usize::from(self.cell_index[cell])
                            ..usize::from(self.cell_index[cell + 1.into()]);
                        for i in range.map(Into::into) {
                            unsafe {
                                self.surround_buffer_x.push(self.x1[i] + offset_x);
                                self.surround_buffer_y.push(self.y1[i] + offset_y);
                                self.surround_buffer_k.push(self.k[i]);
                            }
                        }
                    }*/
                }
                #[cfg(feature = "simd")]
                {
                    for _ in 0..EXTRA_ELEMS {
                        unsafe {
                            self.surround_buffer_x.push(0.0);
                            self.surround_buffer_y.push(0.0);
                        }
                    }
                }
                #[cfg(feature = "simd")]
                const EXTRA_ELEMS: usize = 7; // round down num iterations to (hopefully?) skip these.
                for i in (usize::from(self.cell_index[cell])
                    ..usize::from(self.cell_index[cell + 1.into()]))
                    .map(Into::into)
                {
                    let x0 = self.x0[i];
                    let y0 = self.y0[i];
                    let x1 = self.x1[i];
                    let y1 = self.y1[i];
                    let k = self.k[i];
                    let mut acc_x;
                    let mut acc_y;

                    // about 42 % faster
                    #[cfg(feature = "simd")]
                    unsafe {
                        use simd::*;
                        use std::arch::x86_64::*;
                        let mut acc_x_s = set1(0.0);
                        let mut acc_y_s = set1(0.0);

                        let x1 = set1(x1);
                        let y1 = set1(y1);
                        // 256 = 8xf32 => 8 iterations
                        // step_by rounds down
                        let elements = self.surround_buffer_x.len;
                        let k_base: __m256 =
                            _mm256_load_ps(self.relation_table.as_ptr().add(k as _) as _);
                        for i in (0..elements).step_by(8) {
                            // load ks, zero extend, and index into base
                            let k: __m256 = _mm256_permutevar8x32_ps(
                                k_base,
                                _mm256_cvtepu8_epi32(_mm256_castsi256_si128(loadiu(
                                    self.surround_buffer_k.inner.as_mut_ptr().add(i)
                                        as *const __m256i,
                                ))),
                            );
                            let x_other = load(self.surround_buffer_x.as_ptr().add(i));
                            let y_other = load(self.surround_buffer_y.as_ptr().add(i));

                            // TODO: FMA/fast inv sqrt tricks
                            let dx = sub(x1, x_other);
                            let dy = sub(y1, y_other);
                            let dot = fmadd(dx, dx, mul(dy, dy));
                            let rsqrt = rsqrt_fast(dot);

                            // NOTE: also test _mm256_rcp_ps
                            let r = mul(rsqrt, dot); //sqrt(dot);
                            let r_inv = mul(rsqrt, rsqrt);
                            let x = mul(r, set1(1.0 / POINT_MAX_RADIUS));

                            // do not inline, will change later
                            // 4.0 * (1.0 - (x * (1.0 / BETA))) = (4.0 - (x * (4.0 / BETA)))
                            //let c = mul(set1(4.0), sub(set1(1.0), mul(x, set1(1.0 / BETA))));
                            //let c = fnmadd(x, set1(4.0 / BETA), set1(4.0));
                            let c =
                                fnmadd(r, set1((4.0 / BETA) * (1.0 / POINT_MAX_RADIUS)), set1(4.0));
                            let f = max(
                                mul(
                                    k,
                                    max(min(sub(x, set1(BETA)), sub(set1(1.0), x)), set1(0.0)),
                                ),
                                c,
                            );
                            // > 0.0 will also test for Nan
                            //let mask_ok = _mm256_cmp_ps(dot, set1(0.0), _CMP_GT_OQ);
                            let mask_ok = _mm256_cmp_ps(dot, set1(0.0), _CMP_NEQ_OQ);

                            // f / r
                            let f_div_r = and(mul(f, rsqrt), mask_ok);

                            acc_x_s = fmadd(dx, f_div_r, acc_x_s);
                            acc_y_s = fmadd(dy, f_div_r, acc_y_s);
                        }
                        acc_x = hadd(acc_x_s);
                        acc_y = hadd(acc_y_s);
                    }

                    // about 52 % of execution time is spent here
                    #[cfg(not(feature = "simd"))]
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
                            let k: f32 = *unsafe { local_k_arr.get_unchecked(k_other as usize) };
                            let c = 4.0 * (1.0 - x / BETA);
                            //let c = 10.0 * (1.0 / x - 1.0 / (BETA));

                            let f = ((k / (1.0 - BETA)) * (x - BETA).min(1.0 - x).max(0.0)).max(c);
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
                    debug_assert_float!(acc_x);
                    debug_assert_float!(acc_y);
                    acc_x *= MIN_WIDTH;
                    acc_y *= MIN_WIDTH;
                    debug_assert_float!(acc_x);
                    debug_assert_float!(acc_y);
                    let mut vx = x1 - x0;
                    let mut vy = y1 - y0;

                    for dx in [-WIDTH_X, WIDTH_X] {
                        let c = x1 - x0 + dx;
                        if c.abs() < vx.abs() {
                            vx = c
                        }
                    }
                    for dy in [-WIDTH_Y, WIDTH_Y] {
                        let c = y1 - y0 + dy;
                        if c.abs() < vy.abs() {
                            vy = c
                        }
                    }
                    let v2 = vx * vx + vy * vy;
                    let vx_norm = vx / v2.sqrt();
                    let vy_norm = vy / v2.sqrt();
                    let friction_force = v2 * (50.0 / (DT * DT));
                    if !(vx_norm.is_nan() || vy_norm.is_nan()) {
                        acc_x += friction_force * (-vx_norm);
                        acc_y += friction_force * (-vy_norm);
                    }

                    let new_x = mod_simulation_x(x1 + vx + acc_x * DT * DT); //.rem_euclid(1.0);
                    let new_y = mod_simulation_y(y1 + vy + acc_y * DT * DT); //.rem_euclid(1.0);
                    if !(debug_check_valid_pos(x1)
                        && debug_check_valid_pos(y1)
                        && debug_check_valid_pos(new_x)
                        && debug_check_valid_pos(new_y))
                    {
                        panic!("ERR: x1: {x1}, y1: {y1}, vx: {vx}, vy: {vy}, acc_x: {acc_x}, acc_y: {acc_y} new_x: {new_x}, new_y: {new_y}");
                    }

                    self.x1_tmp[i] = new_x;
                    self.y1_tmp[i] = new_y;

                    self.x0_tmp[i] = x1;
                    self.y0_tmp[i] = y1;

                    let new_cell = compute_cell(new_x, new_y);
                    self.point_cell[i] = new_cell;
                    self.point_cell_offset[i] = self.cell_count[new_cell];
                    self.cell_count[new_cell] += 1.into();
                }
            }
        }

        // The last passes are very serial, but is fast enough
        // that it basically does not matter

        let mut acc: Pid = 0.into();
        for i in (0..CELLS_PLUS_1_U).map(Into::into) {
            self.cell_index[i] = acc;
            let count = replace(&mut self.cell_count[i], 0.into());
            acc += count;
        }

        for i in (0..NUM_POINTS_U).map(Into::into) {
            let cell = self.point_cell[i];
            let index = self.cell_index[cell] + self.point_cell_offset[i];
            self.x0[index] = self.x0_tmp[i];
            self.y0[index] = self.y0_tmp[i];
            self.x1[index] = self.x1_tmp[i];
            self.y1[index] = self.y1_tmp[i];
            self.k_tmp[index] = self.k[i];
        }
        swap(&mut self.k, &mut self.k_tmp);
    }
    fn serialize_gpu(&self, data: &mut Vec<GpuPoint>) {
        data.clear();
        data.extend(
            self.x0
                .iter()
                .zip(self.y0.iter())
                .zip(self.k.iter())
                .map(|((&x, &y), &k)| GpuPoint {
                    x,
                    y,
                    k: k as _,
                    _unused: 0,
                }),
        )
    }
}

/// PPA and offset by 1.
fn ppa_offset_1(cell_count_or_index_new: &mut Cbox<Pid>) {
    let mut acc: Pid = 0.into();
    for i in (0..CELLS_PLUS_1_U).map(Into::into) {
        let curr = cell_count_or_index_new[i];
        cell_count_or_index_new[i] = acc;
        acc += curr;
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
            }
        }
    }};
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let (event_loop, window) = window_setup();

    let mut state = State::new(window).await;
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, window_id } if window_id == state.window.id() => match event {
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
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: k,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                let p = match k {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                };
                use VirtualKeyCode::*;

                match keycode {
                    W => Some(&mut state.keystate.up),
                    S => Some(&mut state.keystate.down),
                    A => Some(&mut state.keystate.left),
                    D => Some(&mut state.keystate.right),
                    Space => Some(&mut state.keystate.zoom_out),
                    LShift => Some(&mut state.keystate.zoom_in),
                    _ => None,
                }
                .map(|e| *e = p);
            }
            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => state.resize(*new_inner_size),
            _ => state.input(&event),
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(SurfaceError::Lost) => state.resize(state.size),
                Err(SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            state.window().request_redraw();
        }
        _ => {}
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

    let window_size = winit::dpi::LogicalSize {
        width: 512,
        height: 512,
    };
    let window = WindowBuilder::new()
        //.with_inner_size(window_size)
        //.with_max_inner_size(window_size)
        //.with_min_inner_size(window_size)
        .build(&event_loop)
        .unwrap();

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

    params_buffer: wgpu::Buffer,
}

impl State {
    async fn new(window: Window) -> Self {
        //let simulation_state = SimulationState::new(SimulationInputEuler::new().verlet());
        let mut simulation_controller = SimThreadController::new();

        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
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
                    features: wgpu::Features::empty(),
                    limits,
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
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            //present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let mut point_transfer_buffer = Vec::new();
        //simulation_state.serialize_gpu(&mut point_transfer_buffer);
        simulation_controller.get_state(&mut point_transfer_buffer);
        let gpu_point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Buffer"),
            contents: bytemuck::cast_slice(&point_transfer_buffer),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let params = Params::new();
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Param Buffer"),
            contents: bytemuck::cast_slice(&bytemuck::cast::<_, [u8; size_of::<Params>()]>(params)),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        });

        let bind_group_layout: wgpu::BindGroupLayout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Default::default(),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: Default::default(),
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        gpu_point_buffer.as_entire_buffer_binding(),
                    ),
                },
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
            .replace("{NUM_POINTS}", &format!("{}", NUM_POINTS))
            .replace("{NUM_KINDS}", &format!("{}", NUM_KINDS))
            .replace("{WIDTH_X}", &format!("{:.10}", WIDTH_X))
            .replace("{WIDTH_Y}", &format!("{:.10}", WIDTH_Y));

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
            let make_primitive = |topology| wgpu::PrimitiveState {
                topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            };
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
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: make_primitive(wgpu::PrimitiveTopology::TriangleList),
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
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        {
            let control_dt = 0.03;

            let k = &self.keystate;

            let lr = k.right as i32 as f32 - k.left as i32 as f32;
            let ud = k.up as i32 as f32 - k.down as i32 as f32;
            let io = k.zoom_out as i32 as f32 - k.zoom_in as i32 as f32;
            self.params.offset_x += lr * control_dt / self.params.zoom_x;
            self.params.offset_y += ud * control_dt / self.params.zoom_x;
            self.params.zoom_x += io * control_dt * self.params.zoom_x;
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
        self.queue.write_buffer(
            &self.gpu_point_buffer,
            0,
            bytemuck::cast_slice(&self.point_transfer_buffer),
        );
        self.simulation_controller
            .try_get_state(&mut self.point_transfer_buffer);
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
        render_pass.draw(0..(3 * NUM_POINTS * 9), 0..1);
        drop(render_pass);

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update(&mut self) {}

    fn input(&mut self, _event: &WindowEvent<'_>) {}

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
    #[inline(always)]
    pub unsafe fn load(a: *const f32) -> __m256 {
        _mm256_load_ps(a)
    }
    pub unsafe fn loadu(a: *const f32) -> __m256 {
        _mm256_loadu_ps(a)
    }
    pub unsafe fn loadi(a: *const __m256i) -> __m256i {
        _mm256_load_si256(a)
    }
    pub unsafe fn loadiu(a: *const __m256i) -> __m256i {
        _mm256_loadu_si256(a)
    }
    #[inline(always)]
    pub unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn add(a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn andi(a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }
    #[inline(always)]
    pub unsafe fn and(a: __m256, b: __m256) -> __m256 {
        _mm256_and_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn andn(a: __m256, b: __m256) -> __m256 {
        _mm256_andnot_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn sub(a: __m256, b: __m256) -> __m256 {
        _mm256_sub_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn fmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    pub unsafe fn fmsub(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    pub unsafe fn fnmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fnmadd_ps(a, b, c)
    }
    #[inline(always)]
    pub unsafe fn fnmsub(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fnmsub_ps(a, b, c)
    }
    #[inline(always)]
    pub unsafe fn gather<const SCALE: i32>(p: *const f32, indexes: __m256i) -> __m256 {
        _mm256_i32gather_ps::<SCALE>(p, indexes)
    }
    #[inline(always)]
    pub unsafe fn sqrt(a: __m256) -> __m256 {
        _mm256_sqrt_ps(a)
    }
    #[inline(always)]
    pub unsafe fn div(a: __m256, b: __m256) -> __m256 {
        _mm256_div_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn rsqrt_fast(a: __m256) -> __m256 {
        _mm256_rsqrt_ps(a)
    }
    #[inline(always)]
    pub unsafe fn set1(a: f32) -> __m256 {
        _mm256_set1_ps(a)
    }
    #[inline(always)]
    pub unsafe fn splati(a: i32) -> __m256i {
        _mm256_set1_epi32(a)
    }
    #[inline(always)]
    pub unsafe fn max(a: __m256, b: __m256) -> __m256 {
        _mm256_max_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn min(a: __m256, b: __m256) -> __m256 {
        _mm256_min_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn slli<const IMM: i32>(a: __m256i) -> __m256i {
        _mm256_slli_epi32::<IMM>(a)
    }
    #[inline(always)]
    pub unsafe fn hadd(a: __m256) -> f32 {
        let mut b: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(b.as_mut_ptr(), a);
        ((b[0] + b[1]) + (b[2] + b[3])) + ((b[4] + b[5]) + (b[6] + b[7]))
    }

    pub fn as_rchunks<T, const N: usize>(this: &[T]) -> (&[T], &[[T; N]]) {
        unsafe fn as_chunks_unchecked<T, const N: usize>(s: &[T]) -> &[[T; N]] {
            let new_len = s.len() / N;
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), new_len) }
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
