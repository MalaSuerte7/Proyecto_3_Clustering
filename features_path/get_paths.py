import os 

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


# # val_features --> paths
# # path --> C:\Users\lifeg\OneDrive\Escritorio\heavy_data\val_subset
# val_videos = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\heavy_data\\val_subset"
# out_val = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\val_paths.txt"
# ## quizas también sin necesidad de ponerlo en un txt se podía hacer pero para visualizarlo mejor que se escriba.
# allpaths_val = getPaths(val_videos)
# pegar_txt(allpaths_val, out_val)


# # train_features --> paths
# # path --> C:\Users\lifeg\OneDrive\Escritorio\heavy_data\train_subset 
# train_videos = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\heavy_data\\train_subset"
# out_train = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\train_paths.txt"
# ## quizas también sin necesidad de ponerlo en un txt se podía hacer pero para visualizarlo mejor que se escriba.
# allpaths_train = getPaths(train_videos)
# pegar_txt(allpaths_train, out_train)


# test_features --> paths
# path --> C:\Users\lifeg\OneDrive\Escritorio\heavy_data\test_subset 
test_videos = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\heavy_data\\test_subset"
out_test = "C:\\Users\\lifeg\\OneDrive\\Escritorio\\Machine\\Proyecto_3_Clustering\\test_paths.txt"
## quizas también sin necesidad de ponerlo en un txt se podía hacer pero para visualizarlo mejor que se escriba.
allpaths_test = getPaths(test_videos)
pegar_txt(allpaths_test, out_test)