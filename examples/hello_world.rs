use egui::{Color32, ColorImage, Context, TextureId, vec2};
use grr::{BlendChannel, BlendFactor, BlendOp, ClearAttachment, ColorBlend, ColorBlendAttachment, Framebuffer, Region, Viewport};
use raw_gl_context::{GlConfig, GlContext, Profile};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use egui::panel::Side;
use egui_grr::{Painter, PainterSettings};


fn main() -> anyhow::Result<()> {
    unsafe {
        let event_loop = EventLoop::new();

        let window = WindowBuilder::new()
            .with_title("egui")
            .with_inner_size(LogicalSize::new(1024.0, 768.0))
            .build(&event_loop)?;

        let context = GlContext::create(
            &window,
            GlConfig {
                version: (4, 6),
                profile: Profile::Core,
                red_bits: 8,
                blue_bits: 8,
                green_bits: 8,
                alpha_bits: 0,
                depth_bits: 0,
                stencil_bits: 0,
                samples: None,
                srgb: true,
                double_buffer: true,
                vsync: true,
            },
        )
            .unwrap();

        context.make_current();

        let mut ctx = Context::default();

        let mut winit = egui_winit::State::new(2048, &window);

        let grr = grr::Device::new(
            |symbol| context.get_proc_address(symbol) as *const _,
            grr::Debug::Enable {
                callback: |report, _, _, _, msg| {
                    println!("{:?}: {:?}", report, msg);
                },
                flags: grr::DebugReport::FULL,
            },
        );

        let pixels : Vec<u8> = image::open("examples/grr.png").unwrap().to_rgba8().to_vec();


        let mut painter = Painter::new(&grr, PainterSettings::default())?;

        let mut first_paint = true;

        let mut name = String::new();

        let mut age = 0;

        let img = ColorImage::from_rgba_unmultiplied([454, 302], &pixels);

        let handle = ctx.load_texture("grr-image", img);



        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::WindowEvent {
                    event,
                    ..
                } => {
                    winit.on_event(&ctx, &event);

                }
                Event::LoopDestroyed => {
                    painter.free(&grr);
                }
                Event::RedrawRequested(_) => {
                    {

                        grr.set_viewport(0, &[Viewport{
                            x: 0.0,
                            y: 0.0,
                            w: window.inner_size().width as _,
                            h: window.inner_size().height as _,
                            n: 0.0,
                            f: 1.0
                        }]);

                        grr.set_scissor(0, &[Region {
                            x: 0,
                            y: 0,
                            w: window.inner_size().width as _,
                            h: window.inner_size().height as _,
                        }]);
                        grr.clear_attachment(Framebuffer::DEFAULT, ClearAttachment::ColorFloat(0, [1.0, 1.0, 1.0, 1.0]));

                        let input = winit.take_egui_input(&window);

                        let mut output = ctx.run(input, |ctx|{
                            egui::SidePanel::new(Side::Left, "side_panel").show(ctx, |ui|{
                                ui.heading("My egui Application");
                                ui.horizontal(|ui| {
                                    ui.label("Your name: ");
                                    ui.text_edit_singleline(&mut name);
                                });
                                ui.add(egui::Slider::new(&mut age, 0..=120).text("age"));
                                if ui.button("Click each year").clicked() {
                                    age += 1;
                                }
                                ui.label(format!("Hello '{}', age {}", name, age));
                                ui.image(handle.id(), vec2(handle.size()[0] as _, handle.size()[1] as _));
                            });
                        });
                        if first_paint{

                            first_paint = !first_paint;
                        }

                        winit.handle_platform_output(&window, &ctx, output.platform_output);


                        let meshes = ctx.tessellate(output.shapes);
                        // draw things behind egui here



                        painter.paint(&grr, ctx.pixels_per_point(), window.inner_size().into(), meshes, output.textures_delta);

                        // draw things on top of egui here



                        context.swap_buffers();


                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                _ => (),
            }
        });
    }
}