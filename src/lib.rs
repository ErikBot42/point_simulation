#![feature(portable_simd)]
use std::marker::PhantomData;
use std::mem::{replace, size_of, swap};
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const CELLS_X: u16 = 128;
const CELLS_Y: u16 = 128;
const CELLS: u16 = CELLS_X * CELLS_Y;
const DT: f32 = 0.00000; // TODO: define max speed/max dt from cell size.
fn compute_cell(x: f32, y: f32) -> Cid {
    // x in 0_f32..1_f32
    // y in 0_f32..1_f32
    let cell_x = ((x * CELLS_X as f32) as u16).min(CELLS_X);
    let cell_y = ((y * CELLS_Y as f32) as u16).min(CELLS_Y);
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

// v0 and v1 will be indexed randomly with gather instruction.
// we want to read x and y simultaneously
//
// register(indexes to v0)
// -> register(xs), register(ys)
// -> reduction
// -> single scalar write to v1
//
// we can perform two 64 bit gathers and interleave the results.
// i0, i1, i2, i3
// i4, i5, i6, i7
//
// ->
// (x0 y0) (x1 y1) (x2 y2) (x3 y3)
// (x4 y4) (x5 y5) (x6 y6) (x7 y7)
// ->
// x0 x1 x2 x3 x4 x5 x6 x7
// y0 y1 y2 y3 y4 y5 y6 y7
//
//
/*
DEFINE INTERLEAVE_DWORDS(src1[127:0], src2[127:0]) {
    dst[31:0] := src1[31:0]
    dst[63:32] := src2[31:0]
    dst[95:64] := src1[63:32]
    dst[127:96] := src2[63:32]
    RETURN dst[127:0]
}
dst[127:0] := INTERLEAVE_DWORDS(a[127:0], b[127:0])
dst[255:128] := INTERLEAVE_DWORDS(a[255:128], b[255:128])
dst[MAX:256] := 0

src1 = x0 y0 x1 y1
src2 = x4 y5 x6 y6


actual order of values is mostly irrelevant

all indices are 32-bit even if elements are 64-bit

-> no need to do interleave tricks with the indexes, just shift upper half (probably specific instruction for this though)


avaliable shuffling instructions:

_mm256_blend_ps/vblendps ymm, ymm, ymm, imm8: (0.33/0.33, 1*p015)
select based on imm8 between two registers. essentially just a mask operation
this but variable exists but is high latency

_mm256_permute_ps/vpermilps ymm, ymm, imm8: (1.0/1.0, 1*p5)
within each 128-bit group, perform arbitrary permute.

_mm256_permutevar8x32_ps/vpermps ymm, ymm, ymm: (1.0/1.0, 1*p5)
generic 32-bit shuffle based on dynamic 32-bit indexes.

strategies:
circular shift r1 by 1 and use blend ops to produce reg with xs and ys

x0 y0 x1 y1 x2 y2 x3 y3
x4 y4 x5 y5 x6 y6 x7 y7

x0    x1    x2    x3
   y0    y1    y2    y3

x4    x5    x6    x7
   y4    y5    y6    y7

x0    x1    x2    x3
y0    y1    y2    y3

   x4    x5    x6    x7
   y4    y5    y6    y7

x0 x4 x1 x5 x2 x6 x3 x7
y0 y4 y1 y5 y2 y6 y3 y7

x0 y0 x1 y1 x2 y2 x3 y3
x4 y4 x5 y5 x6 y6 x7 y7

x0 x1 x2 x3 x4 x5 x6 x7
y0 y1 y2 y3 y4 y5 y6 y7

x0 y0 x1 y1 x2 y2 x3 y3
x4 y4 x5 y5 x6 y6 x7 y7

x0 x2 x4 x6 y0 y2 y4 y6
x1 x3 x5 x7 y1 y3 y5 y7
-> deinterleave ->
x0 x1 x2 x3 x4 x5 x6 x7
y0 y1 y2 y3 y4 y5 y6 y7

instead of doing a gather:
just store point *type* and xy location. ids no longer matter. access pattern will be ideal since writes after accumulation will be localized.

It may be possible to embed type in the point itself (trade mem with compute).
An open question now is if AoS or SoA should be used. => num points and point types can be dynamic.

IEEE-754 32 bit floats:
1 bit sign
8 bit exponent
23 bit fraction

only one time bounds check calculations care about the actual play-field.

sign, exponent:
in 0-1:
0, [0, 127] -> 2 bits unused

in 1-2:
0, [127, 128] -> 8 bits of information unused, 7 relatively packed.

in 2-4:
0, [128, 129] -> 6 packed unused bits -> 32 point types, 12 total -> 4096 types.


16-bit fixed point:
0.0 (0) <-> 1.0 (65535)
This produces 512 possible points within a cell in a 128x128 grid.

expand/compress to f32?
fast sqrt needed for distance calc.



spatial hash -> linear:

struct IterationInfo {
    neighborhood: &[(T;T)],
    this: &[(T;T)]
}


hash table strategies:

I want to scale to 65536 points, with a 128x128 grid.
assume 8 bytes per point/object (2 floats).

naive preallocate:
2d linear array where each cell has a 65536 element vector => 1 GiB + a bunch of associativity conflicts.

paper 1: https://www.researchgate.net/publication/277870601
A Hash Table Construction Algorithm for Spatial Hashing Based on Linear
Memory

calc cell for each point.
struct Cell {
    points: usize,
    start: usize,
    end: usize,
}

all point data is then stored in an array.


improvement to paper 1:
use a Csr

struct SpaceHash {
    counts: Vec<usize>, // # = |cells|
    indexes: Vec<usize> // # = |cells|
    data: Vec<T>,
}
// given original constraints
struct SpaceHash {
    counts: Vec<u16>, // # = |cells|
    indexes: Vec<u16> // # = |cells|
    data: Vec<(f32, f32)>,
}
MEM(n) = cells * 4 + points * 8
scratch(n) = points * 10

this can be improved by replacing counts with indexes in a single pass
struct SpaceHash {
    counts_or_indexes: Vec<u16>, // # = |cells|
    data: Vec<(f32, f32)>,
}
MEM(n) = cells * 2 + points * 8
scratch(n) = points * 10


still need a 2 step process though.

benefit of a Csr style is that SIMD ops can be more linear

dumping all of this into a linearly processable array is trivial

(aaaaaBBBBcccccc
ddd(EEE)fffff
ggggHHHiiiii)
kkkkLLLLmmmm

the linearly processable array will use 3x the memory though


problem: linear processable array wastes memory
solution: write backwards if enough space

aaaAAAaaa
bbbBBBbbb // clear and start writing eeee backwards
cccCCCccc
dddDDDddd


TODO: persistent counts and use same count for both pairs of the iteration? - do later maybe


the buffer would need to be |points| * 4 length, but that should be fine.


Store separate buffers for x and y, deinterleave requires a ton of needless instructions.
It may be beneficial to offset the allocations of the x and y buffers so there are no cache conflicts between them when iterating linearly.

TODO: need previous position somehow

 * */
//
//
// TODO: maybe merge position and kind for ideal cache properties.
// TODO: make this allocation clever and external.
/*
// spacehash_indexes -> tmpstate -> spacehash_counts -> spacehash_indexes
struct SpaceHash {
    counts_or_indexes: Vec<u16>, // |cells|
    data_x: Vec<f32>,            // |points|
    data_y: Vec<f32>,            // |points|
    data_prev_x: Vec<f32>,       // |points|
    data_prev_y: Vec<f32>,       // |points|
    kind: Vec<u8>,               // |points|
}
struct TmpState {
    cell: Vec<u16>,              // |points|
    data_x: Vec<f32>,            // |points|
    data_y: Vec<f32>,            // |points|
    data_prev_x: Vec<f32>,       // |points|
    data_prev_y: Vec<f32>,       // |points|
    kind: Vec<u8>,               // |points|
}
*/

// if we keep indexes into the previous iteration instead:
struct SpaceHash {
    counts_or_indexes: Vec<u16>, // |cells|
    data_x: Vec<f32>,            // |points|
    data_y: Vec<f32>,            // |points|
    prev_data_x: Vec<f32>,       // |points|
    prev_data_y: Vec<f32>,       // |points|
    kind: Vec<u8>,               // |points|
}
struct TmpState {
    cell: Vec<u16>, // |points|
}

// the randomly generated parts of the simulation
struct SimulationInputEuler {
    x: Pbox<f32>,
    y: Pbox<f32>,
    vx: Pbox<f32>,
    vy: Pbox<f32>,
    k: Pbox<u8>,
    relation_table: Box<[f32; KINDS2]>,
}
struct SimulationInputVerlet {
    x0: Pbox<f32>,
    y0: Pbox<f32>,
    x1: Pbox<f32>,
    y1: Pbox<f32>,
    k: Pbox<u8>,
    relation_table: Box<[f32; KINDS2]>,
}
impl SimulationInputEuler {
    fn new() -> SimulationInputEuler {
        let mut rng = Prng(3141592);
        let xy_range = 0.0..1.0;
        let v_range = -1.0..1.0;
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
            relation_table: iter_alloc(|| rng.f32(-1.0..1.0)),
        }
    }
    fn verlet(self) -> SimulationInputVerlet {
        let x1 = self.x;
        let y1 = self.y;
        let mut x0 = self.vx;
        let mut y0 = self.vy;
        for (((x0, y0), x1), y1) in x1
            .iter()
            .zip(y1.iter())
            .zip(x0.iter_mut())
            .zip(y0.iter_mut())
        {
            (*x1, *y1) = (x0 + *x1 * DT, y0 + *y1 * DT);
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
        #[derive(Copy, Clone, Default, Debug)]
        struct $name($index);
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
        &self.0[index.into()]
    }
}
impl<I: Into<usize>, T, const SIZE: usize> IndexMut<I> for Tb<I, T, SIZE> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index.into()]
    }
}
type Pbox<T> = Tb<Pid, T, NUM_POINTS_U>;
type Cbox<T> = Tb<Cid, T, CELLS_PLUS_1_U>;

const NUM_POINTS_U: usize = NUM_POINTS as usize;
const CELLS_PLUS_1_U: usize = CELLS as usize + 1;
const KINDS2: usize = NUM_KINDS as usize * NUM_KINDS as usize;
/// Entire state of the simulation
struct SimulationState {
    /// x value for previous position (shuffled)
    /// overwritten during update.
    x0: Pbox<f32>,
    /// y value for previous position (shuffled)
    /// overwritten during update.
    y0: Pbox<f32>,

    /// x value for new position
    x1: Pbox<f32>,
    /// y value for new position
    y1: Pbox<f32>,

    /// scratch space for x position before being put in right spot
    x_tmp: Pbox<f32>,
    /// scratch space for y position before being put in right spot
    y_tmp: Pbox<f32>,

    /// point kind (not shuffled)
    k_old: Pbox<u8>,
    /// New point kind scratch space.
    k_new: Pbox<u8>,

    /// Map new indexes to old indexes.
    back: Pbox<Pid>,

    /// Scratch space for x position of surrounding points.
    surround_buffer_x: Vec<f32>,
    /// Scratch space for y position of surrounding points.
    surround_buffer_y: Vec<f32>,
    /// Scratch space for kind of surrounding points.
    surround_buffer_k: Vec<u8>,

    /// What cell the new points belong to.
    point_cell_location: Pbox<Cid>,
    /// What number the point has in the cell.
    point_cell_offset: Pbox<Pid>,
    /// How many old points are in a cell OR
    /// what index into the old point buffer this cell has.
    cell_count_or_index: Cbox<Pid>,
    /// How many new points are in a cell OR
    /// what index into the new point buffer this cell has.
    cell_count_or_index_new: Cbox<Pid>,
    /// Mapping kinds to amount of attraction.
    relation_table: Box<[f32; KINDS2]>,
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

fn calloc<T: Default, const SIZE: usize, O>() -> O
where
    O: From<Box<[T; SIZE]>>,
{
    O::from(unsafe {
        (0..SIZE)
            .map(|_| T::default())
            .collect::<Vec<T>>()
            .try_into()
            .unwrap_unchecked()
    })
}
impl SimulationState {
    fn new(input: SimulationInputVerlet) -> Self {
        let SimulationInputVerlet {
            x0,
            y0,
            x1,
            y1,
            k,
            relation_table,
        } = input;
        let mut this = Self {
            x0,
            y0,
            x1: calloc(),
            y1: calloc(),
            x_tmp: calloc(),
            y_tmp: calloc(),
            k_old: k,
            k_new: calloc(),
            back: calloc(),
            surround_buffer_x: Vec::new(),
            surround_buffer_y: Vec::new(),
            surround_buffer_k: Vec::new(),
            point_cell_location: calloc(),
            point_cell_offset: calloc(),
            cell_count_or_index: calloc(),
            cell_count_or_index_new: calloc(),
            relation_table,
        };
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let x_next = x1[i];
            let y_next = y1[i];
            let new_cell = compute_cell(x_next, y_next);
            this.point_cell_location[i] = new_cell;
            let new_offset = this.cell_count_or_index_new[new_cell];
            this.cell_count_or_index_new[new_cell] += 1.into();
            this.point_cell_offset[i] = new_offset;
            this.x_tmp[i] = x_next;
            this.y_tmp[i] = y_next;
        }
        this
    }
    // update starts right after previous physics update to reduce construction size.
    fn update(&mut self) {
        // PPA
        ppa_offset_1(&mut self.cell_count_or_index_new);
        // Insertion.
        {
            for i_old in (0..NUM_POINTS_U).map(Into::into) {
                let i_new = self.cell_count_or_index_new[self.point_cell_location[i_old]]
                    + self.point_cell_offset[i_old];
                self.x0[i_new] = self.x_tmp[i_old];
                self.y0[i_new] = self.y_tmp[i_old];
                self.k_new[i_new] = self.k_old[i_old];
                self.back[i_new] = i_old as _;
            }
        };
        // Swapping vectors.
        swap(&mut self.x0, &mut self.x1);
        swap(&mut self.y0, &mut self.y1);
        swap(&mut self.k_old, &mut self.k_new);
        swap(
            &mut self.cell_count_or_index,
            &mut self.cell_count_or_index_new,
        );

        // <- at this point state is valid

        self.cell_count_or_index_new
            .iter_mut()
            .for_each(|e| *e = 0.into());
        for cell_y in 0..CELLS_Y {
            for cell_x in 0..CELLS_X {
                let cell = Cid(cell_y * CELLS_Y + cell_x);
                self.surround_buffer_x.clear();
                self.surround_buffer_y.clear();
                self.surround_buffer_k.clear();
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let cell = ((((cell_y as i32 + dy + CELLS_Y as i32) % CELLS_Y as i32)
                            * CELLS_X as i32
                            + ((cell_x as i32 + dx + CELLS_X as i32) % CELLS_X as i32))
                            as usize)
                            .into();
                        for i in (usize::from(self.cell_count_or_index[cell])
                            ..usize::from(self.cell_count_or_index[cell + 1.into()]))
                            .map(Into::into)
                        {
                            self.surround_buffer_x.push(self.x1[i]);
                            self.surround_buffer_y.push(self.y1[i]);
                            self.surround_buffer_k.push(self.k_old[i]);
                        }
                    }
                }
                for i in (usize::from(self.cell_count_or_index[cell])
                    ..usize::from(self.cell_count_or_index[cell + 1.into()]))
                    .map(Into::into)
                {
                    let x = self.x1[i];
                    let y = self.y1[i];
                    let k = self.k_new[i];
                    let x_prev = self.x0[self.back[i]];
                    let y_prev = self.y0[self.back[i]];
                    let mut acc_x = 0.0;
                    let mut acc_y = 0.0;
                    for ((other_x, other_y), &other_k) in self
                        .surround_buffer_x
                        .iter()
                        .zip(self.surround_buffer_y.iter())
                        .zip(self.surround_buffer_k.iter())
                    {
                        let diff_x = x - other_x;
                        let diff_y = y - other_y;
                        let r2 = diff_x * diff_x + diff_y * diff_y;

                        let invr = 1.0 / r2.sqrt();

                        let dx = diff_x * invr;
                        let dy = diff_y * invr;

                        let force = self.relation_table
                        [(k as u16 + NUM_KINDS as u16 * other_k as u16) as usize]
                        * 0.01 // TODO: move DT, this factor into relation table.
                        * invr
                        * invr;

                        if dx.is_finite() && dy.is_finite() && force.is_finite() {
                            acc_x += dx * force;
                            acc_y += dy * force;
                        }
                    }
                    if !acc_x.is_finite() {
                        acc_x = 0.0;
                    }
                    if !acc_y.is_finite() {
                        acc_x = 0.0;
                    }
                    let x_next = x_prev.rem_euclid(0.999); //(x+acc_x*0.00001).rem_euclid(0.9999);//let x_next = (x + (x - x_prev) * 0.95 + acc_x * DT * DT).rem_euclid(0.9999);
                    let y_next = y_prev.rem_euclid(0.999); //(y+acc_y*0.00001).rem_euclid(0.9999);//let y_next = (y + (y - y_prev) * 0.95 + acc_y * DT * DT).rem_euclid(0.9999);

                    {
                        let new_cell = compute_cell(x_next, y_next);
                        self.point_cell_location[i] = new_cell;
                        let new_offset = self.cell_count_or_index_new[new_cell];
                        self.cell_count_or_index_new[new_cell] += 1.into();
                        self.point_cell_offset[i] = new_offset;
                        self.x_tmp[i] = x_next;
                        self.y_tmp[i] = y_next;
                    };
                }
            }
        }
    }
    fn serialize_gpu(&self, data: &mut Vec<GpuPoint>) {
        data.clear();
        data.extend(
            self.x0
                .iter()
                .zip(self.y0.iter())
                .zip(self.k_old.iter())
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

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use std::array::from_fn;

use prng::Prng;
use vertex::{Vertex, VERTICES};

mod prng;
mod vertex;

const NUM_POINTS: u32 = 2048;
const NUM_KINDS: usize = 8;
const HASH_TEXTURE_SIZE: u32 = 256;
const RANGE_INDEX: u32 = 8;
const BETA: f64 = 0.4;
const RADIUS_CLIP_SPACE: f64 = (RANGE_INDEX as f64) * 1.0 / (HASH_TEXTURE_SIZE as f64);
const INNER_RADIUS_CLIP_SPACE: f64 = BETA * RADIUS_CLIP_SPACE;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Instant, SystemTime};

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

#[derive(Debug)]
struct SimParams {
    forces: Vec<f32>,
}

impl SimParams {
    fn new() -> SimParams {
        let mut rng;
        //rng = Prng(13);
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                rng = Prng(13);
            } else {
                rng = Prng(SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64);
            }
        }

        printlnc!("seed: {}", rng.0);

        SimParams {
            forces: (0..NUM_KINDS * NUM_KINDS)
                .map(|_| rng.f32(-1.0..1.0))
                .collect(),
        }
    }
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
                        state: ElementState::Pressed,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                use VirtualKeyCode::*;
                match keycode {
                    U => state.use_simd = dbgc!(!state.use_simd),
                    _ => (),
                }
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
        width: 540,
        height: 540,
    };
    let window = WindowBuilder::new()
        .with_inner_size(window_size)
        .with_max_inner_size(window_size)
        .with_min_inner_size(window_size)
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
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,

    bind_groups: [wgpu::BindGroup; 2],

    point_textures: [wgpu::Texture; 2],

    #[cfg(not(target_arch = "wasm32"))]
    last_print: Instant,
    frame: usize,

    sim_params: SimParams,

    use_simd: bool,

    point_read_buffer: wgpu::Buffer,
    point_write_buffer: wgpu::Buffer,
    simulation_state: SimulationState,

    point_transfer_buffer: Vec<GpuPoint>,
    gpu_point_buffer: wgpu::Buffer,
}

impl State {
    async fn new(window: Window) -> Self {
        let simulation_state = SimulationState::new(SimulationInputEuler::new().verlet());

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

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: dbgc!(limits),
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
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let num_vertices = VERTICES.len() as u32;

        let point_textures: [wgpu::Texture; 2] = from_fn(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Point Texture"),
                size: wgpu::Extent3d {
                    width: NUM_POINTS,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[wgpu::TextureFormat::Rgba32Float],
            })
        });

        let mut point_transfer_buffer = Vec::new();
        simulation_state.serialize_gpu(&mut point_transfer_buffer);
        let gpu_point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&point_transfer_buffer),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let point_read_buffer = {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Point Buffer"),
                size: NUM_POINTS as u64 * (4 * size_of::<f32>() as u64),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let point_write_buffer = {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Point Buffer"),
                size: NUM_POINTS as u64 * (4 * size_of::<f32>() as u64),
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let point_texture_views: [wgpu::TextureView; 2] = point_textures
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let sim_params = SimParams::new();

        let bind_group_layout: wgpu::BindGroupLayout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: Default::default(),
                        },
                        count: None,
                    },
                ],
            });

        let bind_groups: [_; 2] = point_texture_views
            .iter()
            .map(|point_texture_view| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(point_texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(
                                gpu_point_buffer.as_entire_buffer_binding(),
                            ),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let shader_source: String = std::fs::read_to_string("src/shader.wgsl")
            .unwrap()
            .replace("{NUM_POINTS}", &format!("{}", NUM_POINTS))
            .replace("{NUM_KINDS}", &format!("{}", NUM_KINDS));

        dbgc!(RADIUS_CLIP_SPACE);

        let render_pipeline;
        let fill_pipeline;
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
            let vertex_fill = wgpu::VertexState {
                module: &fragment_vertex_shader,
                entry_point: "vs_fill",
                buffers: &[Vertex::desc()],
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
            fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Fill Pipeline"),
                layout,
                vertex: vertex_fill.clone(),
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_fill",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Float,
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

        // init state
        {
            let mut fill_encoder: wgpu::CommandEncoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Fill Encoder"),
                });
            let mut fill_pass = fill_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fill Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &point_texture_views[0],
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
            fill_pass.set_pipeline(&fill_pipeline);
            fill_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            fill_pass.set_bind_group(0, &bind_groups[1], &[]);
            fill_pass.draw(0..num_vertices, 0..1);
            drop(fill_pass);
            queue.submit(Some(fill_encoder.finish()));
        }

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            bind_groups,
            #[cfg(not(target_arch = "wasm32"))]
            last_print: Instant::now(),
            frame: 0,
            sim_params,
            point_read_buffer,
            point_write_buffer,
            point_textures,
            use_simd: true,
            simulation_state,
            point_transfer_buffer,
            gpu_point_buffer,
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
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

        let mut encoder: wgpu::CommandEncoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder"),
                });

        {
            encoder.copy_texture_to_buffer(
                self.point_textures[0].as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: &self.point_read_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(size_of::<[f32; 4]>() as u32 * NUM_POINTS),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: NUM_POINTS,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );

            self.queue.submit(Some(encoder.finish()));
            self.queue.write_buffer(
                &self.gpu_point_buffer,
                0,
                bytemuck::cast_slice(&self.point_transfer_buffer),
            );
            static mut FOO: u8 = 0;
            if unsafe {
                FOO = (FOO + 1) % 30;
                FOO == 0
            } {
                self.simulation_state.update();
            }
            self.simulation_state
                .serialize_gpu(&mut self.point_transfer_buffer);
            encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder"),
                });

            let read_buffer_slice = self.point_read_buffer.slice(..);
            read_buffer_slice.map_async(wgpu::MapMode::Read, |_| ());
            self.device.poll(wgpu::Maintain::Wait);
            let mut data: Vec<EulerPoint> =
                bytemuck::cast_slice(&read_buffer_slice.get_mapped_range()).to_vec();

            update_points(&mut data, &self.sim_params);

            let write_buffer_slice = self.point_write_buffer.slice(..);
            write_buffer_slice.map_async(wgpu::MapMode::Write, |_| ());
            self.device.poll(wgpu::Maintain::Wait);
            bytemuck::cast_slice_mut(&mut write_buffer_slice.get_mapped_range_mut())
                .copy_from_slice(&data);

            self.point_read_buffer.unmap();
            self.point_write_buffer.unmap();

            encoder.copy_buffer_to_texture(
                wgpu::ImageCopyBuffer {
                    buffer: &self.point_write_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(size_of::<[f32; 4]>() as u32 * NUM_POINTS),
                        rows_per_image: Some(1),
                    },
                },
                self.point_textures[0].as_image_copy(),
                wgpu::Extent3d {
                    width: NUM_POINTS,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }

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
        render_pass.set_bind_group(0, &self.bind_groups[0], &[]);
        render_pass.draw(0..(3 * NUM_POINTS), 0..1);
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

const GRID_SIZE: usize = 24;
const GRID_LEN: f32 = 2.0 / GRID_SIZE as f32;
const GRID_RADIUS: f32 = GRID_LEN / 2.0;

#[inline(never)]
fn update_points(points: &mut [EulerPoint], params: &SimParams) {
    #[inline(always)]
    fn hash_point_pos(px: f32, py: f32) -> (isize, isize) {
        let px = ((px + 1.0) / 2.0).rem_euclid(1.0);
        let py = ((py + 1.0) / 2.0).rem_euclid(1.0);
        ((px * GRID_SIZE as f32) as _, (py * GRID_SIZE as f32) as _)
    }
    #[inline(always)]
    fn join_coords((px, py): (isize, isize)) -> usize {
        ((px.rem_euclid(GRID_SIZE as isize))
            + (py.rem_euclid(GRID_SIZE as isize)) * GRID_SIZE as isize) as usize
    }

    let relation_lookup: [[f32; NUM_KINDS]; NUM_KINDS] =
        from_fn(|i| from_fn(|j| params.forces[i + NUM_KINDS * j] * -0.1));

    let mut spatial_hash: Vec<Vec<u32>> =
        (0..(GRID_SIZE * GRID_SIZE)).map(|_| Vec::new()).collect();

    let mut outer_group: Vec<u32> = Vec::with_capacity(NUM_POINTS as _);

    for _ in 0..5 {
        for e in spatial_hash.iter_mut() {
            e.clear();
        }

        for i in 0..NUM_POINTS as usize {
            let p = &points[i];
            let key = join_coords(hash_point_pos(p.x, p.y));
            spatial_hash[key].push(i as u32);
        }

        let dt = 0.05;

        let grid_size = GRID_SIZE as isize;
        for (hash_index, inner_group) in spatial_hash.iter().enumerate() {
            outer_group.clear();

            for jj in [
                -grid_size - 1,
                -grid_size,
                -grid_size + 1,
                -1,
                0,
                1,
                grid_size - 1,
                grid_size,
                grid_size + 1,
            ]
            .into_iter()
            .map(|d| {
                &spatial_hash[(d + hash_index as isize).rem_euclid(grid_size * grid_size) as usize]
            }) {
                outer_group.extend_from_slice(jj);
            }

            evaluate_groups(inner_group, &outer_group, points, relation_lookup, dt);
        }
    }
}

fn evaluate_groups(
    inner_group: &[u32],
    outer_group: &[u32],
    points: &mut [EulerPoint],
    relation_lookup: [[f32; NUM_KINDS]; NUM_KINDS],
    dt: f32,
) {
    let (outer_remaining, outer_chunks) = as_rchunks::<_, NUM_KINDS>(outer_group);
    for &i in inner_group {
        if true {
            unsafe {
                let p_x = points.get_unchecked(i as usize).x;
                let p_y = points.get_unchecked(i as usize).y;

                let mut p_vx = points.get_unchecked(i as usize).vx;
                let mut p_vy = points.get_unchecked(i as usize).vy;

                let mut p_vx_s = _mm256_setzero_ps();
                let mut p_vy_s = _mm256_setzero_ps();

                for outer_chunk in outer_chunks {
                    let j_s: __m256i = _mm256_loadu_si256(outer_chunk.as_ptr() as _);

                    let points_ptr: *const f32 = points.as_ptr() as _;

                    let x2_s: __m256 = gather::<4>(points_ptr.offset(0), slli::<2>(j_s));
                    let y2_s: __m256 = gather::<4>(points_ptr.offset(1), slli::<2>(j_s));

                    let dix_s: __m256 = sub(splat(p_x), x2_s);
                    let diy_s: __m256 = sub(splat(p_y), y2_s);

                    let r2_s = fmadd(dix_s, dix_s, mul(diy_s, diy_s));
                    let r_inv_s = rsqrt(r2_s);
                    let r_s = mul(r2_s, r_inv_s);

                    let dixn_s = mul(r_inv_s, dix_s);
                    let diyn_s = mul(r_inv_s, diy_s);

                    let x_s = mul(r_s, splat(1.0 / GRID_RADIUS));
                    let beta = 0.3;
                    let relation_force_s = gather::<4>(
                        (relation_lookup.as_ptr() as *const f32)
                            .add((i as usize % NUM_KINDS) * NUM_KINDS),
                        andi(j_s, splati(0b111)),
                    );
                    assert_eq!(NUM_KINDS, 8, "0x111");
                    let f_s = relation_force_s;
                    let fac_dt_s = mul(
                        max(
                            mul(
                                max(min(sub(x_s, splat(beta)), sub(splat(1.0), x_s)), splat(0.0)),
                                f_s,
                            ),
                            sub(splat(1.0), mul(x_s, splat(1.0 / beta))),
                            //fmsub(x_s, splat(1.0 / beta), splat(1.0))
                        ),
                        splat(dt),
                    );
                    let ok_mask = _mm256_cmp_ps::<0x8>(fac_dt_s, splat(0.0));
                    let fac_dt_masked_s = andn(ok_mask, fac_dt_s);

                    let fac_dt_s = fac_dt_masked_s;

                    let final_vx = mul(dixn_s, fac_dt_s);
                    let final_vy = mul(diyn_s, fac_dt_s);
                    let ok_mask_x = _mm256_cmp_ps::<0x8>(final_vx, splat(0.0));
                    let ok_mask_y = _mm256_cmp_ps::<0x8>(final_vy, splat(0.0));
                    let final_vx = andn(ok_mask_x, final_vx);
                    let final_vy = andn(ok_mask_y, final_vy);

                    p_vx_s = add(final_vx, p_vx_s);
                    p_vy_s = add(final_vy, p_vy_s);
                }

                {
                    let mut buffer: [f32; 8] = [0.0; 8];
                    _mm256_storeu_ps(buffer.as_mut_ptr() as _, p_vx_s);
                    let diff = buffer.into_iter().sum::<f32>();
                    p_vx += diff;

                    let mut buffer: [f32; 8] = [0.0; 8];
                    _mm256_storeu_ps(buffer.as_mut_ptr() as _, p_vy_s);
                    let diff = buffer.into_iter().sum::<f32>();
                    p_vy += diff;
                }

                for &j in outer_remaining {
                    let p2 = points[j as usize];
                    let p = points[i as usize];

                    let (dix, diy) = (p.x - p2.x, p.y - p2.y);
                    let r2 = dix * dix + diy * diy;
                    let r = r2.sqrt();
                    let (dixn, diyn) = (dix / r, diy / r);
                    let fac = {
                        let x = r / GRID_RADIUS;
                        let beta = 0.3;
                        let relation_force =
                            relation_lookup[i as usize % NUM_KINDS][j as usize % NUM_KINDS];
                        let f = relation_force;

                        f32::max(
                            f32::max(f32::min(x - beta, -x + 1.0), 0.0) * f,
                            1.0 - x / beta,
                        )
                    };

                    let dx = fac * dixn * dt;
                    let dy = fac * diyn * dt;
                    p_vx += if dx.is_nan() { 0.0 } else { dx };
                    p_vy += if dy.is_nan() { 0.0 } else { dy };
                }
                points[i as usize].vx = p_vx;
                points[i as usize].vy = p_vy;
            }
        } else {
            for &j in outer_group {
                // !0: we know that i != j here and that positions are distinct
                // 0: i may equal j, r may equal zero

                let p2 = points[j as usize];
                let p = &mut points[i as usize];

                let (dix, diy) = (p.x - p2.x, p.y - p2.y);
                let r2 = dix * dix + diy * diy;
                let r = r2.sqrt();
                let (dixn, diyn) = (dix / r, diy / r);
                let fac = {
                    let x = r / GRID_RADIUS;
                    let beta = 0.3;
                    let relation_force =
                        relation_lookup[i as usize % NUM_KINDS][j as usize % NUM_KINDS];
                    let f = relation_force;

                    f32::max(
                        f32::max(f32::min(x - beta, -x + 1.0), 0.0) * f,
                        1.0 - x / beta,
                    )
                };

                let dx = fac * dixn * dt;
                let dy = fac * diyn * dt;
                p.vx += if dx.is_nan() {
                    //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                    0.0
                } else {
                    dx
                };
                p.vy += if dy.is_nan() {
                    //(i as f32 / NUM_POINTS as f32 * 0.5) * 0.0001
                    0.0
                } else {
                    dy
                };
            }
        }

        let p = &mut points[i as usize];

        if p.x > 1.0 {
            p.x = -1.0;
            //p.vx = -p.vx.abs() - dt;
        } else if p.x < -1.0 {
            p.x = 1.0;
            //p.vx = p.vx.abs() - dt;
        }
        if p.y > 1.0 {
            p.y = -1.0;
            //p.vy = -p.vy.abs() - dt;
        } else if p.y < -1.0 {
            p.y = 1.0;
            //p.vy = p.vy.abs() + dt;
        }

        p.vx *= 0.01_f32.powf(dt);
        p.vy *= 0.01_f32.powf(dt);

        p.vx -= p.x * (dt * 0.001);
        p.vy -= p.y * (dt * 0.001);

        p.x += p.vx * dt;
        p.y += p.vy * dt;
    }
}

use core::arch::x86_64::*;
use simd::*;
mod simd {
    use core::arch::x86_64::*;
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
    pub unsafe fn gather<const SCALE: i32>(p: *const f32, indexes: __m256i) -> __m256 {
        _mm256_i32gather_ps::<SCALE>(p, indexes)
    }
    #[inline(always)]
    pub unsafe fn rsqrt(a: __m256) -> __m256 {
        _mm256_rsqrt_ps(a)
    }
    #[inline(always)]
    pub unsafe fn splat(a: f32) -> __m256 {
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
