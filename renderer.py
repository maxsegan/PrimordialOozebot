import bpy
import os
import json
import bmesh
from mathutils import Vector
from math import radians

filepath = '/output/robo1256.txt'

class VIEW3D_PT_creature_viz(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Animation"
    bl_label = 'Blenderer'

    def draw(self, context):
        #obj = context.active_object

        layout = self.layout
        col = layout.column(align=True)
        #col.prop(context.scene, 'creature_path')
        row = layout.row()
        row.operator("blenderer.import")

class ANIM_OT_import_creature(bpy.types.Operator):
    bl_label = "Import"
    bl_idname = "blenderer.import"
    bl_description = "Imports an oozebot"
    bl_options = {'REGISTER', 'UNDO'}

    def invoke(self, context, event):
        self.execute(context)
        return {'FINISHED'}

    def execute(op, context):
        obj = context.active_object
        dirname = os.path.dirname(__file__)
        filename = dirname[:-len('blenderer.blend/')] + filepath
        with open(filename) as json_file:
            data = json.load(json_file)

            vertices = []
            edges = []
            faces = []

            for vertex in data["masses"]:
                vertices.append((vertex[0], vertex[2], vertex[1]))
            for edge in data["springs"]:
                edges.append((edge[0], edge[1]))
            ## TODO!!! figure out what this whole mesh thing is and translate masses
            # Oh and faces
            name = filepath[len('/output/'):]
            mesh = bpy.data.meshes.new(name="{} Mesh".format(name))
            mesh.from_pydata(vertices, edges, faces)
            mesh.update()
            mesh.validate()

            obj = bpy.data.objects.new(name, mesh)
            scene = bpy.context.scene
            scene.collection.objects.link(obj)
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(state=True)
            context.view_layer.objects.active = obj
            print("\nCREATURE VIZ:{}".format(str(context.area.type)))

            # Animate
            bpy.context.window_manager.animall_properties.key_points = True

            mesh = obj.data
            vertices = mesh.vertices
            frame_ctr = 0
            bpy.ops.object.mode_set(mode='OBJECT')
            for frame in data["simulation"]:
                bm = bmesh.new()
                bm.from_mesh(mesh)

                bpy.context.scene.frame_set(frame_ctr)

                for i, v in enumerate(bm.verts):
                    v.co = Vector((frame[i][0], frame[i][2], frame[i][1]))
                # Finish up, write the bmesh back to the mesh
                bm.to_mesh(mesh)
                bm.free()  # free and prevent further access
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.anim.insert_keyframe_animall()
                bpy.ops.object.mode_set(mode='OBJECT')
                frame_ctr += 1
            obj.rotation_euler = (radians(90), 0, 0)


def register():
    bpy.utils.register_class(VIEW3D_PT_creature_viz)
    bpy.utils.register_class(ANIM_OT_import_creature)


def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_creature_viz)
    bpy.utils.unregister_class(ANIM_OT_import_creature)


if __name__ == "__main__":
    register()