#![feature(portable_simd)]

use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

const DELTA_TIME: f32 = 0.05;

use std::mem::{replace, size_of};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct EulerPoint {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}
struct EulerRepr {
    points: Vec<EulerPoint>,
}


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








 * */
//
struct VerletRepr {
    v0: Vec<f32>, // x | y
    v1: Vec<f32>, // x | y
}

//fn euler_to_verlet(euler: &EulerRepr, dt: f32) -> VerletRepr {
//    let mut x0 = Vec::new();
//    let mut y0 = Vec::new();
//    let mut x1 = Vec::new();
//    let mut y1 = Vec::new();
//    for point in &euler.points {
//        x0.push(point.x);
//        y0.push(point.y);
//        x1.push(point.x + point.vx * dt);
//        y1.push(point.y + point.vy * dt);
//    }
//    VerletRepr { x0, y0, x1, y1 }
//}
//fn verlet_to_euler(verlet: &VerletRepr, dt: f32) -> EulerRepr {
//    let mut points = Vec::new();
//    for (((&x0, &y0), &x1), &y1) in verlet
//        .x0
//        .iter()
//        .zip(verlet.y0.iter())
//        .zip(verlet.x1.iter())
//        .zip(verlet.y1.iter())
//    {
//        points.push(EulerPoint {
//            x: x0,
//            y: y0,
//            vx: (x1 - x0) / dt,
//            vy: (y1 - y0) / dt,
//        });
//    }
//    EulerRepr { points }
//}

unsafe impl bytemuck::Zeroable for EulerPoint {}
unsafe impl bytemuck::Pod for EulerPoint {}

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
}

impl State {
    async fn new(window: Window) -> Self {
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

        let limits = wgpu::Limits::downlevel_webgl2_defaults();

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
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let bind_groups: [_; 2] = point_texture_views
            .iter()
            .map(|point_texture_view| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind group"),
                    layout: &bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(point_texture_view),
                    }],
                })
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let shader_source: String = include_str!("shader.wgsl").to_owned();
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
