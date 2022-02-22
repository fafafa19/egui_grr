#version 460 core

layout (location = 0) uniform vec2 u_screen_size;

layout (location = 0) in vec2 v_pos;
layout (location = 1) in vec2 v_tc;
layout (location = 2) in uvec4 v_srgba;



layout (location = 0) out vec4 a_srgba;
layout (location = 1) out vec2 a_tc;

// 0-1 linear  from  0-255 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb), srgba.a / 255.0);
}

void main() {
    gl_Position = vec4(
                      2.0 * v_pos.x / u_screen_size.x - 1.0,
                      1.0 - 2.0 * v_pos.y / u_screen_size.y,
                      0.0,
                      1.0);
    // egui encodes vertex colors in gamma spaces, so we must decode the colors here:
    a_srgba = linear_from_srgba(v_srgba);
    a_tc = v_tc;
}