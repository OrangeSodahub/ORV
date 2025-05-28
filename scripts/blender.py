import bpy
import bpy_extras
from mathutils import Vector
import numpy as np

def get_scene_inverse_depth_range(scene, cam):
    """Calculate inverse depth range (2% and 98% percentiles) for all vertices"""
    inverse_depths = []
    
    for obj in scene.objects:
        if obj.type != 'MESH':
            continue
            
        world_matrix = obj.matrix_world
        mesh = obj.data
        
        for vertex in mesh.vertices:
            world_coord = world_matrix @ vertex.co
            distance = (world_coord - cam.matrix_world.translation).length
            inverse_depth = 1.0 / distance if distance > 0 else 0
            inverse_depths.append(inverse_depth)
    
    if not inverse_depths:
        return 0, 1  # default values
        
    inverse_depths = np.array(inverse_depths)
    min_inverse = np.percentile(inverse_depths, 2)
    max_inverse = np.percentile(inverse_depths, 98)
    
    return min_inverse, max_inverse

def set_vertex_colors_for_object(obj, scene, cam, inverse_depth_range):
    """Set vertex colors and material for a single object"""
    if obj.type != 'MESH':
        return
    
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y
    
    mesh = obj.data
    if obj.data.shape_keys or obj.modifiers:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        object_eval = obj.evaluated_get(depsgraph)
        mesh_eval = object_eval.data
    else:
        mesh_eval = mesh
    
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="Col")
    
    color_layer = mesh.vertex_colors.active
    world_matrix = obj.matrix_world
    cam_location = cam.matrix_world.translation
    cam_direction = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    
    min_inverse, max_inverse = inverse_depth_range
    inverse_range = max_inverse - min_inverse
    
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            world_coord = world_matrix @ mesh_eval.vertices[vertex_index].co
            to_vertex = (world_coord - cam_location).normalized()
            is_backface = to_vertex.dot(cam_direction) < 0
            screen_coord = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_coord)
            
            distance = (world_coord - cam_location).length
            inverse_depth = 1.0 / distance if distance > 0 else 0
            
            b = (inverse_depth - min_inverse) / inverse_range if inverse_range > 0 else 0
            b = max(0, min(1, b))
            
            r = screen_coord.x
            g = 1 - screen_coord.y
            
            if r < 0 or r > 1 or g < 0 or g > 1 or is_backface:
                r, g, b = 0, 0, 0
            
            color_layer.data[loop_index].color = (r, g, b, 1.0)
    
    # Setup material
    mat_name = f"ScreenSpaceColor_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    vertex_color = nodes.new(type='ShaderNodeVertexColor')
    vertex_color.layer_name = "Col"
    
    emission = nodes.new(type='ShaderNodeEmission')
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    links = mat.node_tree.links
    links.new(vertex_color.outputs[0], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs[0])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def process_all_objects():
    """Process all mesh objects in the scene"""
    scene = bpy.context.scene
    cam = scene.camera
    
    if not cam:
        print("No camera in scene!")
        return
    
    inverse_depth_range = get_scene_inverse_depth_range(scene, cam)
    
    for obj in scene.objects:
        if obj.type == 'MESH':
            print(f"Processing object: {obj.name}")
            set_vertex_colors_for_object(obj, scene, cam, inverse_depth_range)

process_all_objects() 