import os 

out = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\answer.txt"
videos_folder_choice = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\testing_videos"

def getPaths(videos_folder, valido = ['mp4']):
    paths = []
    # sabiendo que la captera tiene puros videos
    list_videos = os.listdir(videos_folder)

    for file in list_videos:
        file_path = os.path.join(videos_folder, file)

        # .mp4 
        if os.path.isfile(file_path) and any(file.endswith(vd) for vd in valido):
            paths.append(file_path)
    return paths

def pegar_txt(paths , output_txt):
    with open(output_txt, 'w') as file:
        for path in paths:
            file.write(path + '\n')

allpaths = getPaths(videos_folder_choice)
pegar_txt(allpaths, out)