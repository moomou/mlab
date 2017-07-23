from collections import OrderedDict
import h5py


def open_multi(names, mode='a', driver=None, comm=None):
    print('Opening::', driver, comm)

    output = OrderedDict()
    for filename in names:
        output[filename] = h5py.File(filename, mode, driver=driver, comm=comm)

    return output


def close_multi(handles):
    for _, h in handles.items():
        try:
            h.close()
        except Exception as e:
            print(e, flush=True)


def open_one(name, mode='a', driver=None, comm=None):
    return h5py.File(name, mode=mode, driver=driver, comm=comm)
