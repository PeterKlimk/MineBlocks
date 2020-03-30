extern crate winit;
extern crate vulkano;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate cgmath;
extern crate glyph_brush;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;
use vulkano::format::Format;
use vulkano::image::{AttachmentImage, ImageUsage, ImmutableImage, Dimensions};
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use vulkano_win::VkSurfaceBuild;

use winit::{EventsLoop, Window, WindowBuilder, Event, WindowEvent, DeviceEvent};

use std::sync::Arc;

use glyph_brush::{BrushAction, BrushError, GlyphBrushBuilder, Section};

use std::collections::HashMap;

use cgmath::{InnerSpace, SquareMatrix};

use image::{ImageFormat};

#[derive(Default, Debug, Clone)]
struct Vertex { 
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn new(x: f32, y: f32, z: f32, u: f32, v: f32) -> Self {
        Vertex {
            position: [x, y, z],
            tex_coords: [u, v],
        }
    }

    fn new_cast(x: usize, y: usize, z: usize, u: f32, v: f32) -> Self {
        Vertex {
            position: [x as f32, y as f32, z as f32],
            tex_coords: [u, v],
        }
    }
}

vulkano::impl_vertex!(Vertex, position, tex_coords);

#[derive(PartialEq, Eq, Hash)]
struct Position {
    x: i32,
    y: i32,
    z: i32,
}

#[derive(Clone, Copy)]
enum BlockType {
    Dirt,
    Water,
    Air,
}


#[derive(Clone, Copy)]
struct Block {
    block_type: BlockType,
}


#[derive(Clone)]
enum MeshType {
    Regular,
    Water
}

#[derive(Clone)]
struct Mesh {
    vertex_buffer: Arc<ImmutableBuffer<[Vertex]>>,
    mesh_type: MeshType,
}

const CHUNK_SIZE: usize = 16;

type Blocks = [Block; CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE];

#[derive(Clone)]
struct Chunk {
    blocks: Blocks,
    meshes: Option<Vec<Mesh>>,
}

const BLOCK_USAGE: [usize; 36] = [
    0, 2, 1, //back (z: 0)
    0, 3, 2,
    5, 4, 7, //front (z: 1)
    5, 7, 6, 
    0, 7, 4, //left (x:0)
    0, 4, 3, 
    1, 2, 5, //right (x:1)
    1, 5, 6, 
    0, 6, 7, //top (y:0)
    0, 1, 6,
    2, 3, 4, //bottom (y:1)
    2, 4, 5, 
];

const SIDE_OFFSET: [isize; 6] = [
    -(CHUNK_SIZE as isize),
    CHUNK_SIZE as isize,
    -1,
    1,
    -((CHUNK_SIZE * CHUNK_SIZE) as isize),
    (CHUNK_SIZE * CHUNK_SIZE) as isize,
];

impl Chunk {
    fn new (blocks: Blocks) -> Self {
        Chunk {
            blocks: blocks,
            meshes: None,
        }
    }

    fn meshify(&mut self, queue: Arc<Queue>) -> Box<GpuFuture> {
        let mut regular_vertices = Vec::new();
        let mut water_vertices = Vec::new();

        for pos in 0..(CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE) {
            match self.blocks[pos].block_type {
                BlockType::Air => continue,
                _ => {},
            }
            let x = pos % CHUNK_SIZE;
            let z = (pos / CHUNK_SIZE) % CHUNK_SIZE;
            let y = pos / (CHUNK_SIZE * CHUNK_SIZE); 

            let points: [cgmath::Point3<usize>; 8] = [
                cgmath::Point3::new(0, 0, 0),
                cgmath::Point3::new(1, 0, 0),
                cgmath::Point3::new(1, 1, 0),
                cgmath::Point3::new(0, 1, 0),
                cgmath::Point3::new(0, 1, 1),
                cgmath::Point3::new(1, 1, 1),
                cgmath::Point3::new(1, 0, 1),
                cgmath::Point3::new(0, 0, 1),
            ];

            let mut i = 0;

            while i < BLOCK_USAGE.len() {
                let vi = BLOCK_USAGE[i];
                let side = i / 6;

                if i % 6 == 0 {
                    let offset = SIDE_OFFSET[side];
                    let t = offset.abs() as usize;
                    let bad = match offset.is_negative() {
                        false => CHUNK_SIZE - 1,
                        true  => 0,
                    };

                    if (pos / t) % CHUNK_SIZE != bad {
                        let other_pos = (pos as isize + offset);
                        match self.blocks[other_pos as usize].block_type {
                            BlockType::Dirt => {
                                i += 6;
                                continue;
                            },
                            BlockType::Water => {
                                if let BlockType::Water = self.blocks[pos].block_type {
                                    i += 6;
                                    continue;
                                }
                            }
                            _ => {}
                        }
                    }
                }

                let point = points[vi];
                let abs_point = point + cgmath::Vector3::new(x, y, z);
                let (u, v) = match side {
                    0 => (point.x as f32 / 16.0, point.y as f32 / 16.0),
                    1 => (point.x as f32 / 16.0, point.y as f32 / 16.0),
                    2 => (point.z as f32 / 16.0, point.y as f32 / 16.0),
                    3 => (point.z as f32 / 16.0, point.y as f32 / 16.0),
                    4 => (point.x as f32 / 16.0, point.z as f32 / 16.0),
                    5 => (point.x as f32 / 16.0, point.z as f32 / 16.0),
                    _ => panic!("Invalid Side."),
                };

                let add = Vertex::new_cast(abs_point.x, abs_point.y, abs_point.z, u, v);
                match self.blocks[pos].block_type {
                    BlockType::Water => water_vertices.push(add),
                    _ => regular_vertices.push(add),
                }

                i += 1;
            }

        }

        let (regular_vertex_buffer, future) = ImmutableBuffer::from_iter(regular_vertices.iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();
        let (water_vertex_buffer, future2) = ImmutableBuffer::from_iter(water_vertices.iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();

        let future = future.join(future2);

        self.meshes = Some(vec![
            Mesh {
                vertex_buffer: regular_vertex_buffer,
                mesh_type: MeshType::Regular,
            },
            Mesh {
                vertex_buffer: water_vertex_buffer,
                mesh_type: MeshType::Water,
            }
        ]);

        Box::new(future)
    }
}

type EntityID = u32;

struct Entity {
    entityID: EntityID,
}

struct World {
    currentID: EntityID,
    entities: HashMap<EntityID, Entity>,
}

struct Parts {
    Velocity: HashMap<EntityID, cgmath::Vector3<f32>>,
    Position: HashMap<EntityID, cgmath::Point3<f32>>,
}

fn main() {
    let mut chunks: HashMap<Position, Chunk> = HashMap::new();

    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).unwrap()
    };
    
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    
    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();
    
    let queue_family = physical.queue_families().find(|&q| {
        
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
        [(queue_family, 0.5)].iter().cloned()).unwrap();
    
    let queue = queues.next().unwrap();   

    let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
        let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    } else {
        return;
    }; 
        
    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
            initial_dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
            PresentMode::Fifo, true, None).unwrap()
    };
    
    let atch_usage = ImageUsage {
        transient_attachment: true,
        input_attachment: true,
        ..ImageUsage::none()
    };

    mod base_vert {
        vulkano_shaders::shader!{
            ty: "vertex",
            path: "src/assets/shaders/shad.vert"
        }
    }

    mod base_frag {
        vulkano_shaders::shader!{
            ty: "fragment",
            path: "src/assets/shaders/shad.frag"
        }
    }

    // mod wire_frag {
    //     vulkano_shaders::shader!{
    //         ty: "fragment",
    //         path: "src/assets/shaders/wire.frag"
    //     }
    // }

    mod water_vert {
        vulkano_shaders::shader!{
            ty: "vertex",
            path: "src/assets/shaders/water.vert"
        }
    }

    mod water_frag {
        vulkano_shaders::shader!{
            ty: "fragment",
            path: "src/assets/shaders/water.frag"
        }
    }
    
    let blend = AttachmentBlend::alpha_blending();

    let base_vert = base_vert::Shader::load(device.clone()).unwrap();
    let base_frag = base_frag::Shader::load(device.clone()).unwrap();
    // let wire_frag = wire_frag::Shader::load(device.clone()).unwrap();
    
    let water_vert = water_vert::Shader::load(device.clone()).unwrap();
    let water_frag = water_frag::Shader::load(device.clone()).unwrap();

    let (texture, tex_future) = {
        let image = image::load_from_memory_with_format(include_bytes!("assets/atlas.png"),
            ImageFormat::PNG).unwrap().to_rgba();
        let width = image.width();
        let height = image.height();

        let image_data = image.into_raw().clone();

        ImmutableImage::from_iter(
            image_data.iter().cloned(),
            Dimensions::Dim2d { width: width, height: height},
            Format::R8G8B8A8Srgb,
            queue.clone()
        ).unwrap()
    };

    let mut future = Box::new(tex_future) as Box<dyn GpuFuture>;
    
    for x in -5..6 {
        for z in -5..6 {
            
            let mut blocks = [Block { block_type: BlockType::Dirt }; CHUNK_SIZE*CHUNK_SIZE*CHUNK_SIZE];
            for x in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    blocks[z * CHUNK_SIZE + x] = Block { block_type: BlockType::Water };
                    blocks[CHUNK_SIZE*CHUNK_SIZE + z * CHUNK_SIZE + x] = Block { block_type: BlockType::Water };
                }
            }

            let mut chunk = Chunk::new(blocks);
            future = Box::new(future.join(chunk.meshify(queue.clone())));
            chunks.insert(Position {x: x, y: 0, z: z}, chunk.clone());
        }
    }

    let mut previous_frame_end = future;

    let sampler = Sampler::new(device.clone(), Filter::Nearest, Filter::Nearest,
        MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();
    
    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {                 
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap());

    let water_pipeline = Arc::new(GraphicsPipeline::start()
        .blend_collective(blend.clone())
        .depth_stencil_simple_depth()
        .vertex_input_single_buffer()
        .vertex_shader(water_vert.main_entry_point(), ())    
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(water_frag.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    
    let regular_pipeline = Arc::new(GraphicsPipeline::start()
        .blend_collective(blend.clone())
        .depth_stencil_simple_depth()
        .vertex_input_single_buffer()
        .vertex_shader(base_vert.main_entry_point(), ())    
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(base_frag.main_entry_point(), ())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());
    
    let set = Arc::new(PersistentDescriptorSet::start(regular_pipeline.clone(), 0)
        .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
        .build().unwrap()
    );

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None };

    let depth_buffer = AttachmentImage::with_usage(device.clone(), initial_dimensions, Format::D16Unorm, atch_usage).unwrap();
    let mut framebuffers = window_size_dependent_setup(depth_buffer.clone(), &images, render_pass.clone(), &mut dynamic_state);
    let mut recreate_swapchain = false;

    let mut yaw = 2.8;
    let mut pitch = -0.2;

    let mut px = 0.0;
    let mut py = 0.0;
    let mut pz = 0.0;
    
    let mut mw = false;
    let mut ma = false;
    let mut ms = false;
    let mut md = false;
    
    let mut mouse_inside = true;

    window.grab_cursor(true);
    window.hide_cursor(true);

    loop {
        previous_frame_end.cleanup_finished();
        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            let depth_buffer = AttachmentImage::with_usage(device.clone(), dimensions, Format::D16Unorm, atch_usage).unwrap();
            framebuffers = window_size_dependent_setup(depth_buffer.clone(), &new_images, render_pass.clone(), &mut dynamic_state);
            recreate_swapchain = false;
        }
        
        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };
        
        let clear_values = vec![[0.0, 0.0, 0.0, 0.0].into(), 1.0f32.into()];

        let mat_pitch = cgmath::Matrix4::from_angle_x(cgmath::Rad::<f32>(pitch));
        let mat_yaw = cgmath::Matrix4::from_angle_y(cgmath::Rad::<f32>(yaw));
        
        let rotate = mat_pitch * mat_yaw;

        let mut move_vec = cgmath::Vector4::new(0.0, 0.0, 0.0, 0.0);
        if mw {
            move_vec += cgmath::Vector4::new(0.0, 0.0, 1.0, 0.0);
        }

        if ma {
            move_vec += cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0);
        }

        if ms {
            move_vec += cgmath::Vector4::new(0.0, 0.0, -1.0, 0.0);
        }

        if md {
            move_vec += cgmath::Vector4::new(-1.0, 0.0, 0.0, 0.0);
        }

        move_vec = rotate.invert().unwrap() * move_vec;
        if move_vec.magnitude() > 1.0 {
             move_vec = move_vec.normalize();
        }

        px += move_vec.x;
        py += move_vec.y;
        pz += move_vec.z;

        let view_matrix = rotate * cgmath::Matrix4::from_translation(cgmath::Vector3::new(px, py, pz));

        let dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return;
        };

        let projection_matrix = cgmath::perspective(
            cgmath::Deg::<f32>(90.0),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            1000.0,
        );

        let mut command_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap();

        for (pos, chunk) in &chunks {
            let model_matrix: cgmath::Matrix4<f32> = cgmath::Matrix4::from_translation(cgmath::Vector3::new(
                CHUNK_SIZE as f32 * pos.x as f32, 
                CHUNK_SIZE as f32 * pos.y as f32, 
                CHUNK_SIZE as f32 * pos.z as f32
            ));

            let mvp = projection_matrix * view_matrix * model_matrix;
            let push_constants = base_vert::ty::PushConstantData {
                projection: mvp.into(),
            };

            for mesh in chunk.meshes.as_ref().unwrap() {
                match mesh.mesh_type {
                    MeshType::Regular => {
                        command_builder = command_builder
                            .draw(regular_pipeline.clone(), &dynamic_state, mesh.vertex_buffer.clone(), set.clone(), push_constants)
                            .unwrap();
                    }
                    MeshType::Water => {
                        command_builder = command_builder
                            .draw(water_pipeline.clone(), &dynamic_state, mesh.vertex_buffer.clone(), set.clone(), push_constants)
                            .unwrap();
                    }
                }
            }
        }
        
        command_builder = command_builder
            .end_render_pass()
            .unwrap();
        
        let command_buffer = command_builder.build().unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()

            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }


        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent { event: WindowEvent::KeyboardInput {
                    input: winit::KeyboardInput {state: winit::ElementState::Pressed, virtual_keycode: Some(virtual_code), .. },
                    .. }, ..} => {
                    match virtual_code {
                        winit::VirtualKeyCode::W => { mw = true; }
                        winit::VirtualKeyCode::A => { ma = true; }
                        winit::VirtualKeyCode::S => { ms = true; }
                        winit::VirtualKeyCode::D => { md = true; }
                        _ => {}
                    }
                }

                Event::WindowEvent { event: WindowEvent::KeyboardInput {
                    input: winit::KeyboardInput {state: winit::ElementState::Released, virtual_keycode: Some(virtual_code), .. },
                    .. }, ..} => {
                    match virtual_code {
                        winit::VirtualKeyCode::W => { mw = false; }
                        winit::VirtualKeyCode::A => { ma = false; }
                        winit::VirtualKeyCode::S => { ms = false; }
                        winit::VirtualKeyCode::D => { md = false; }
                        _ => {}
                    }
                }

                Event::WindowEvent { event: WindowEvent::CursorLeft{..}, .. } => mouse_inside = false,
                Event::WindowEvent { event: WindowEvent::CursorEntered{..}, .. } => mouse_inside = true,

                Event::DeviceEvent { event: DeviceEvent::MouseMotion{ delta, .. }, ..} => {
                    if mouse_inside {
                        yaw += (delta.0 / 200.0) as f32;
                        pitch -= (delta.1 / 200.0) as f32;
                        if pitch > std::f32::consts::PI / 2.0 {
                            pitch = std::f32::consts::PI / 2.0;
                        }

                        else if pitch < -std::f32::consts::PI / 2.0 {
                            pitch = -std::f32::consts::PI / 2.0;
                        }
                    }
                }
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => done = true,
                Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
                _ => ()
            }
        });
        if done { return; }
    }
}


fn window_size_dependent_setup(
    depth_buffer: Arc<AttachmentImage>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}