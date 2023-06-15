use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::time::Instant;

const NUM_POINTS: u32 = 512;

use std::mem::size_of;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    pollster::block_on(run());
}
async fn run() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_inner_size(winit::dpi::LogicalSize {
            width: 960,
            height: 540,
        })
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(window).await;
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => match event {
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
            WindowEvent::Resized(physical_size) => state.resize(*physical_size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                state.resize(**new_inner_size)
            }
            _ => state.input(event),
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
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,
    fill_pipeline: wgpu::RenderPipeline,
    update_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,

    bind_groups: (wgpu::BindGroup, wgpu::BindGroup),

    point_texture_views: (wgpu::TextureView, wgpu::TextureView),


    last_print: Instant,
    frame: usize,
}
impl State {
    async fn new(window: Window) -> Self {
        // apply:     pos, vel -> pos, _
        // aggregate: pos+, _ -> _ , vel
        //
        // aggregate + apply: pos0+, vel0 -> pos1, vel1

        // Limits {
        //     max_texture_dimension_1d: 2048,
        //     max_texture_dimension_2d: 2048,
        //     max_texture_dimension_3d: 256,
        //     max_texture_array_layers: 256,
        //     max_bind_groups: 4,
        //     max_bindings_per_bind_group: 640,
        //     max_dynamic_uniform_buffers_per_pipeline_layout: 8,
        //     max_dynamic_storage_buffers_per_pipeline_layout: 0,
        //     max_sampled_textures_per_shader_stage: 16,
        //     max_samplers_per_shader_stage: 16,
        //     max_storage_buffers_per_shader_stage: 0,
        //     max_storage_textures_per_shader_stage: 0,
        //     max_uniform_buffers_per_shader_stage: 11,
        //     max_uniform_buffer_binding_size: 16384,
        //     max_storage_buffer_binding_size: 0,
        //     max_vertex_buffers: 8,
        //     max_buffer_size: 268435456,
        //     max_vertex_attributes: 16,
        //     max_vertex_buffer_array_stride: 255,
        //     min_uniform_buffer_offset_alignment: 256,
        //     min_storage_buffer_offset_alignment: 256,
        //     max_inter_stage_shader_components: 60,
        //     max_compute_workgroup_storage_size: 0,
        //     max_compute_invocations_per_workgroup: 0,
        //     max_compute_workgroup_size_x: 0,
        //     max_compute_workgroup_size_y: 0,
        //     max_compute_workgroup_size_z: 0,
        //     max_compute_workgroups_per_dimension: 0,
        //     max_push_constant_size: 0,
        // }

        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();
        dbg!(instance
            .enumerate_adapters(wgpu::Backends::all())
            .collect::<Vec<_>>());
        let adapter: wgpu::Adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .find(|adapter| adapter.is_surface_supported(&surface))
            .unwrap();
        dbg!(adapter.features());
        let limits = wgpu::Limits::downlevel_webgl2_defaults();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::POLYGON_MODE_LINE
                        | wgpu::Features::POLYGON_MODE_POINT,
                    limits: dbg!(limits),
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
            .filter(|f| f.is_srgb())
            .next()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            //present_mode: wgpu::PresentMode::AutoVsync,
            present_mode: wgpu::PresentMode::Immediate,
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

        // rgba32float

        let make_point_texture = || {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
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
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[wgpu::TextureFormat::Rgba32Float],
            })
        };

        let point_textures = (make_point_texture(), make_point_texture());

        let point_texture_views: (wgpu::TextureView, wgpu::TextureView) = (
            point_textures
                .0
                .create_view(&wgpu::TextureViewDescriptor::default()),
            point_textures
                .1
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );

        let bind_group_layout: wgpu::BindGroupLayout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let bind_groups: (wgpu::BindGroup, wgpu::BindGroup) = (
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&point_texture_views.0),
                }],
            }),
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&point_texture_views.1),
                }],
            }),
        );
        
        const INDEX_OFFSET: usize = 16;
        let shader_source: String = format!("
@group(0) @binding(0) 
var compute_texture: texture_2d<f32>;


struct FillVertexInput {{
    @location(0) position: vec2<f32>,
}}
struct FillVertexOuput {{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) pos: vec4<f32>,
}}

@vertex
fn vs_fill(
    in: FillVertexInput
) -> FillVertexOuput {{
    var out: FillVertexOuput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.pos = vec4<f32>(in.position, 0.0, 1.0);
    return out;
}}

@fragment
fn fs_fill(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {{

    //let a = textureLoad(compute_texture, vec2<u32>(group * 10u, 0u), 0);

    
    let num_points = {NUM_POINTS};
    let c: f32 = in.pos.x;
    return vec4<f32>(c, sin(c * (f32(num_points) + 1.9008)), cos(c*2338.23894), sin(c*2783.7893));
}}

@fragment
fn fs_update(
    in: FillVertexOuput,
) -> @location(0) vec4<f32> {{
    




    var a = textureLoad(compute_texture, vec2<u32>(u32(in.clip_position.x), 0u), 0);
    //let dt = 0.04;
    let dt = 0.004;
    var p = a.xy;
    var v = a.zw;


    //var a = textureLoad(compute_texture, vec2<u32>(u32(in.pos.x * {NUM_POINTS}.0 + 0.5), 0u), 0);
    
    var attract_p = vec2<f32>(0.0);
    var repell_p = vec2<f32>(0.0);
    for (var i: i32 = 0; i < {NUM_POINTS}; i++) {{
        let my_id = u32(in.clip_position.x);
        var f: f32 = 0.0;
        switch(my_id % 7u) {{
            case 0u: {{ 
                switch (u32(i % 7)) {{
                    case 0u: {{ f = -0.9754196802233119; }}
                    case 1u: {{ f = 0.8749759205556629; }}
                    case 2u: {{ f = -0.32970000065553795; }}
                    case 3u: {{ f = -0.5206510756977798; }}
                    case 4u: {{ f = 0.6376401564518979; }}
                    case 5u: {{ f = -0.6083062473168999; }}
                    case 6u: {{ f = -0.27903002956423073; }}
                    default: {{ f = 0.0; }}
                }}
            }}
            case 1u: {{
                switch (u32(i % 7)) {{
                    case 0u: {{ f = 0.9778881484752089; }}
                    case 1u: {{ f = 0.649413257898205; }}
                    case 2u: {{ f = 0.17558277573203918; }}
                    case 3u: {{ f = 0.727256798783394; }}
                    case 4u: {{ f = 0.12817386742495174; }}
                    case 5u: {{ f = -0.6812167941122906; }}
                    case 6u: {{ f = -0.40886325239861354; }}
                    default: {{ f = 0.0; }}
                }}
            }}
            case 2u: {{
                switch (u32(i % 7)) {{
                    case 0u: {{ f = -0.6286519291003476; }}
                    case 1u: {{ f = 0.822062236958949; }}
                    case 2u: {{ f = -0.8371226217691972; }}
                    case 3u: {{ f = 0.4368971281321796; }}
                    case 4u: {{ f = -0.7509632917170683; }}
                    case 5u: {{ f = -0.6142576128932369; }}
                    case 6u: {{ f = -0.9361177593141474; }}
                    default: {{ f = 0.0; }}
                }}
            }}
            case 3u: {{
                switch (u32(i % 7)) {{
                    case 0u: {{ f = -0.504991369048782; }}
                    case 1u: {{ f = 0.8672848328013054; }}
                    case 2u: {{ f = -0.35082800680361115; }}
                    case 3u: {{ f = 0.7514574353943266; }}
                    case 4u: {{ f = -0.19706051563746407; }}
                    case 5u: {{ f = -0.11736513162143702; }}
                    case 6u: {{ f = 0.8032107450780448; }}
                    default: {{ f = 0.0; }}
                }}
            }}
            case 4u: {{
                switch (u32(i % 7)) {{
                    case 0u: {{ f = -0.07457054601772439; }}
                    case 1u: {{ f = -0.4872470355575358; }}
                    case 2u: {{ f = 0.5276985248539052; }}
                    case 3u: {{ f = -0.36851241475518015; }}
                    case 4u: {{ f = 0.7713418014525202; }}
                    case 5u: {{ f = 0.14443692700941058; }}
                    case 6u: {{ f = -0.15613729001780752; }}
                    default: {{ f = 0.0; }}
                }}
            }}
            case 5u: {{
                switch (u32(i % 7)) {{
                    case 0u: {{ f = -0.5892377301220046; }}
                    case 1u: {{ f = 0.5351996056533119; }}
                    case 2u: {{ f = -0.4012452739961294; }}
                    case 3u: {{ f = -0.4830133446690492; }}
                    case 4u: {{ f = -0.9124036949915313; }}
                    case 5u: {{ f = 0.12376897004590304; }}
                    case 6u: {{ f = -0.7530831515583702; }}
                    default: {{ f = 0.0; }}
                }}
            }}
            case 6u: {{
                switch (u32(i % 7)) {{
                    case 0u: {{ f = -0.2707709796782187; }}
                    case 1u: {{ f = 0.618195490989037; }}
                    case 2u: {{ f = -0.407641487578563; }}
                    case 3u: {{ f = 0.7306004972634732; }}
                    case 4u: {{ f = -0.7042080904077299; }}
                    case 5u: {{ f = 0.3478501083148249; }}
                    case 6u: {{ f = 0.9912197039589776; }}
                    default:  {{ f = 0.0; }}
                }}
            }}
            default: {{ f = 0.0; }}
        }}

        var other = textureLoad(compute_texture, vec2<u32>(u32(i), 0u), 0);

        // TODO: let hardware do the wrapping?
        // TODO: rearrange to make sampling cheaper
        //
        if u32(i) != my_id {{
            let diff = (p.xy - other.xy);
            let l = length(diff);

            let r_inner = 0.02;

            let r_outer = r_inner * 4.0;

            //if l < r_inner {{
            //    f = 1.0 / l - 1.0 / r_inner;
            //}} else if todo {{
            //    
            //}} else {{
            //    f = 0.0;
            //}}

            let a = 2.0 * f / (r_outer - r_inner);
            let q = (l - r_inner) * a;
            let u = (-l + r_outer) * a;

            let w = 32.0;

            //let z = (1.0 / l) - (1.0 / r_inner);
            let z = w - (l / r_inner) * w;
            f = max(
                max(
                    0.0,
                    min(q * sign(a), u * sign(a))
                ) * sign(a),
                z
            );

            
            

            //if l > 0.01 {{
            //    f = 0.0;
            //}}
            v += 0.1 * f * normalize(diff) * dt;
        }}
        
        

    }}
    

    //v += 0.04 * -(smoothstep(0.0, 0.3, length(p-attract_p)) - 0.5) * normalize(p-attract_p) * dt * 0.9;
    //v += 0.03 *  (smoothstep(0.0, 0.3, length(p-repell_p)) - 0.5) * normalize(p-repell_p) * dt * 0.9;
    //v += 0.03 *  -(smoothstep(0.0, 1.0, length(p)) - 0.5) * normalize(p) * dt * 2.0;
    v -= p * dot(p, p) * dt * 0.1;


    
    // reflection
    if p.x > 1.0 {{
        v.x = -abs(v.x);
    }} else if p.x < -1.0 {{
        v.x = abs(v.x);
    }}
    if p.y > 1.0 {{
        v.y = -abs(v.y);
    }} else if p.y < -1.0 {{
        v.y = abs(v.y);
    }}

    v *= pow(0.3, dt);

    p += v * dt;
    return vec4<f32>(p, v);
}}

struct VertexInput {{
    //@location(0) position: vec2<f32>,
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
}}

struct VertexOutput {{
    @builtin(position) clip_position: vec4<f32>,
    @location(0) group: u32,
}};

@vertex
fn vs_main(
    in: VertexInput,
) -> VertexOutput {{
    var out: VertexOutput;

    let group = in.vertex_index / 3u;
    let member = in.vertex_index % 3u;
    
    var offset: vec2<f32>;

    switch member {{
        case 0u: {{
            offset = vec2<f32>(sqrt(3.0) / 2.0, -0.5);
        }}
        case 1u: {{
            offset = vec2<f32>(0.0, 1.0);
        }}
        case 2u, default: {{
            offset = vec2<f32>(-sqrt(3.0) / 2.0, -0.5);
        }}
    }}
    offset *= 0.01;
    
    let a = textureLoad(compute_texture, vec2<u32>(group * 1u, 0u), 0);

    offset += a.xy;

    //out.clip_position = vec4<f32>(offset.x + f32(group) * 0.01 - 1.0, offset.y + sin(f32(group)) * 0.5, 0.0, 1.0);
    out.clip_position = vec4<f32>(offset.x, offset.y, 0.0, 1.0);
    out.group = group;
    return out;   
}}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {{
    let a = textureLoad(compute_texture, vec2<u32>(in.group, 0u), 0);
    let q = fract(f32(in.group) / 7.0);
    return vec4<f32>(1.0-q, q, 0.0, 1.0);
}}

");

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
        let primitive: wgpu::PrimitiveState = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
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
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout,
            vertex: wgpu::VertexState {
                module: &fragment_vertex_shader,
                entry_point: "vs_main",
                //buffers: &[Vertex::desc()],
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
            primitive,
            depth_stencil: None,
            multisample,
            multiview: None,
        });
        let targets = &[Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba32Float,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let buffers = &[Vertex::desc()];
        let fill_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fill Pipeline"),
            layout,
            vertex: wgpu::VertexState {
                module: &fragment_vertex_shader,
                entry_point: "vs_fill",
                buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_vertex_shader,
                entry_point: "fs_fill",
                targets,
            }),
            primitive,
            depth_stencil: None,
            multisample,
            multiview: None,
        });
        let update_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Update Pipeline"),
            layout,
            vertex: wgpu::VertexState {
                module: &fragment_vertex_shader,
                entry_point: "vs_fill",
                buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &fragment_vertex_shader,
                entry_point: "fs_update",
                targets,
            }),
            primitive,
            depth_stencil: None,
            multisample,
            multiview: None,
        });

        let mut encoder: wgpu::CommandEncoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fill Encoder"),
            });
        let mut fill_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Fill Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &point_texture_views.0,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.01,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        fill_pass.set_pipeline(&fill_pipeline);
        fill_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        fill_pass.set_bind_group(0, &bind_groups.1, &[]);
        fill_pass.draw(0..num_vertices, 0..1);
        drop(fill_pass);
        queue.submit(Some(encoder.finish()));

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            fill_pipeline,
            update_pipeline,
            vertex_buffer,
            num_vertices,
            bind_groups,
            point_texture_views,
            last_print: Instant::now(),
            frame: 0,
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        self.frame += 1;
        let max_frame = 240;
        if self.frame == max_frame {
            self.frame = 0;
            let dt = std::mem::replace(&mut self.last_print, Instant::now()).elapsed();
            let fps = max_frame as f64 / dt.as_secs_f64();
            println!("fps: {fps:?}, dt: {dt:50?}");
        }

        let output: wgpu::SurfaceTexture = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder: wgpu::CommandEncoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Update Encoder"),
                });
        for _ in 0..10 {
            let mut update_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Update Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.point_texture_views.1,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.01,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            update_pass.set_pipeline(&self.update_pipeline);
            update_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            update_pass.set_bind_group(0, &self.bind_groups.0, &[]);
            update_pass.draw(0..self.num_vertices, 0..1);
            drop(update_pass);

            let mut update_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Update Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.point_texture_views.0,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.01,
                            b: 0.01,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            update_pass.set_pipeline(&self.update_pipeline);
            update_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            update_pass.set_bind_group(0, &self.bind_groups.1, &[]);
            update_pass.draw(0..self.num_vertices, 0..1);
            drop(update_pass);
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.01,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_groups.1, &[]);
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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}
impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// A    D
//
// C    B
#[rustfmt::skip]
const VERTICES: &[Vertex] = {
    const A: Vertex = Vertex { position: [ 1.0,  1.0]};
    const B: Vertex = Vertex { position: [-1.0, -1.0]};
    const C: Vertex = Vertex { position: [ 1.0, -1.0]};
    const D: Vertex = Vertex { position: [-1.0,  1.0]};
    &[
        A, B, C, 
        A, D, B,
    ]
};
