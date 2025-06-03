# -*- coding: utf-8 -*-
import rhinoscriptsyntax as rs
import os
import time

subD_obj_dataset_path = "/Users/tix/itmo/diploma/dataset/ModelNet40"
tmp_folder_path = "/Users/tix/itmo/diploma/dataset/tmp_folder"

def conversion_process(input_obj):
    filename = os.path.basename(input_obj)
    name, extension = os.path.splitext(filename)
    output_file_path = os.path.join(os.path.dirname(input_obj), name + "_subD" + extension)

    rs.DeleteObjects(rs.AllObjects())

    if rs.Command('-_Import "{}" _Enter'.format(input_obj)):
        # print("Импорт {} выполнен успешно!".format(input_obj))

        rs.SelectObjects(rs.AllObjects())
        rs.Command('_ToSubD _Enter', echo=True)
        subd_objs = rs.LastCreatedObjects()

        obj_export_settings = (
            "_Geometry=_Mesh "
            "_ExportMeshTextureCoordinates=_No "
            "_ExportMeshVertexNormals=_No "
            "_ExportMaterialDefinitions=_No "
            "_YUp=_No "
            "_Enter"
        )
        export_command = '-_Export "{}" {}'.format(output_file_path, obj_export_settings)
        rs.SelectObjects(rs.AllObjects())

        if rs.Command(export_command):
            # print("Экспорт в objects выполнен успешно!")
            time.sleep(0.5)
            if os.path.exists(output_file_path):
                os.remove(input_obj)
        else:
            print("ERROR при экспорте файла: {}!".format(output_file_path))
    else:
        print("ERROR при импорте файла: {}!".format(input_obj))
def get_all_paths(subD_obj_dataset_path):
    obj_path = []
    for file in os.listdir(subD_obj_dataset_path):
        if os.path.isdir(os.path.join(subD_obj_dataset_path, file)):
            current_file = os.path.join(subD_obj_dataset_path, file)
            for file1 in os.listdir(current_file):
                current_file1 = os.path.join(current_file, file1)
                if os.path.isdir(current_file1):
                    for file2 in os.listdir(current_file1):
                        obj_path.append(os.path.join(current_file1, file2))
    return obj_path

all_paths = get_all_paths(subD_obj_dataset_path)
total = len(all_paths)
for i, path in enumerate(all_paths, 1):
    print("Обработка {}/{} — {}".format(i, total, os.path.basename(path)))
    conversion_process(path)
    percent = int((i / total) * 100)
    print("[{:<50}] {}%".format("#" * (percent // 2), percent))






