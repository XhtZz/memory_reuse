import caffe
from caffe.proto.caffe_pb2 import NetParameter

class mem_reuse():
    def __init__(self, net_file, net_params):
        self.caffe_net = caffe.Net(net_file, net_params, caffe.TEST)
        with open(net_file, 'r') as fp:
            self.net = NetParameter()
            pb.text_format.Parse(fp.read(), self.net)

    def find_reuse_blob(self, blob_size, blob_name):
            isFind = False
            for (index, size) in self.reuse_size_list:
                if size >= blob_size and self.isFree[index] == True:
                    self.reuse_id[blob_name] = index
                    self.isFree[index] = False
                    isFind = True
                    break
            if not isFind:
                find_index = -1
                find_num = 0
                for (i, (index, size)) in enumerate(self.reuse_size_list):
                    if self.isFree[index]:
                        self.reuse_id[blob_name] = index
                        find_index = index
                        find_num = i
                if find_index != -1:
                    self.reuse_size_list[find_num] = (find_index, blob_size)
                    self.isFree[find_index] = False
                else:
                    self.reuse_size_list.append((self.reuse_list_size, blob_size))
                    self.isFree[self.reuse_list_size] = False
                    self.reuse_id[blob_name] = self.reuse_list_size
                    self.reuse_list_size += 1
            self.reuse_size_list = sorted(self.reuse_size_list, key = lambda x: x[1])

    def MemoryReuse(self):
        layers = self.net.layer
        blobs = self.caffe_net.blobs
        self.layer_num = len(layers)
        blobs_size = []
        self.reuse_id = {}
        self.reuse_list_size = 0
        self.reuse_size_list = []
        refcount_dict = collections.OrderedDict()

        for x in range(self.layer_num):
            if layers[x].type in ['Input', 'ReLU', 'PReLU','TanH', 'Softmax', 'Sigmoid']:
                continue
            blobs_size.append(blobs[layers[x].name].count)
            for bottom in layers[x].bottom:
                if not bottom in refcount_dict:
                    refcount_dict[bottom] = 1
                else:
                    refcount_dict[bottom] += 1

        for x in range(self.layer_num):
            if layers[x].type in ['ReLU', 'TanH', 'PReLU', 'Softmax', 'Sigmoid']:
                continue
            self.find_reuse_blob(blobs[layers[x].name].count, layers[x].name)
            for bottom in layers[x].bottom:
                refcount_dict[bottom] -= 1
                if refcount_dict[bottom] == 0:
                    self.isFree[self.reuse_id[bottom]] = True