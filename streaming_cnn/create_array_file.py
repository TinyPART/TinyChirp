import torch
import torch.nn as nn
from CNNModels import StreamingCNNArch
def write_weights_to_file(model, file_name):
    with open(file_name, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"{name}\n")
            weights = param.data.numpy().flatten()
            print(weights)
            print(name)
            f.write(' '.join(map(str, weights)))
            f.write('\n')

def write_weights_to_header(model, file_name):
    with open(file_name, "w") as f:
        f.write("#ifndef ARRAY_DATA_H\n")
        f.write("#define ARRAY_DATA_H\n")
        f.write("\n")
        for name, param in model.named_parameters():
            weights = param.data.numpy().flatten()
            dim = param.dim()
            shape = param.shape
            if 1 in shape:
                dim -=1
                shape = shape[:shape.index(1)] + shape[shape.index(1) + 1 :]
            name = name.replace(".","")
            
            write_weights(f, dim, weights, shape, name)
        f.write("\n")
        f.write("#endif")



            

def write_weights(file, dim, weights, shape, name):
    if dim == 1:
        file.write(f"static real_t {name}[] = ")
        file.write("{")
        file.write(','.join(map(str, weights)))
        file.write('};')
        file.write('\n')
    if dim == 2:
        for i in range(shape[0]):
            write_weights(file, 1, weights[i*shape[1]: (i+1)*shape[1]],shape,f"_{name}_{i}")
        file.write(f"static real_t* {name}[] = ")
        file.write("{")   
        for i in range(shape[0]):
            file.write(f"_{name}_{i}")
            if i != shape[0] - 1:
                file.write(', ')
        file.write('};')
        file.write('\n')
    if dim == 3:
        for i in range(shape[0]):
            write_weights(file, 2, weights[i*shape[1]*shape[2]: (i+1)*shape[1]*shape[2]],shape[1:] ,f"_{name}_{i}")
        file.write(f"static real_t** {name}[] = ")
        file.write('{')
        for i in range(shape[0]):
            file.write(f"_{name}_{i}")
            if i != shape[0] -1:
                file.write(', ')
        file.write('};')
        file.write('\n')    


if __name__ =='__main__':
    model = StreamingCNNArch()
    model.load_state_dict(torch.load("model.59.0.941964328289032.pth",map_location="cpu"))
    write_weights_to_header(model,"array_data.h")


    
