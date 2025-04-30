
module {
  func.func @main(%arg0: !torch.vtensor<[2,10],si64>) -> !torch.vtensor<[2,10,128],f32> {
    %0 = torch.vtensor.literal(dense_resource<torch_tensor_1000_64_torch.float32> : tensor<1000x64xf32>) : !torch.vtensor<[1000,64],f32>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %1 = torch.aten.embedding %0, %arg0, %int-1, %false, %false_0 : !torch.vtensor<[1000,64],f32>, !torch.vtensor<[2,10],si64>, !torch.int, !torch.bool, !torch.bool -> !torch.vtensor<[2,10,64],f32>
    %2 = torch.vtensor.literal(dense_resource<torch_tensor_128_64_torch.float32> : tensor<128x64xf32>) : !torch.vtensor<[128,64],f32>
    %3 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %4 = torch.aten.linear %1, %2, %3 : !torch.vtensor<[2,10,64],f32>, !torch.vtensor<[128,64],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %str = torch.constant.str "none"
    %5 = torch.aten.gelu %4, %str : !torch.vtensor<[2,10,128],f32>, !torch.str -> !torch.vtensor<[2,10,128],f32>
    %6 = torch.vtensor.literal(dense_resource<torch_tensor_128_128_torch.float32> : tensor<128x128xf32>) : !torch.vtensor<[128,128],f32>
    %7 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32_1> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %8 = torch.aten.linear %5, %6, %7 : !torch.vtensor<[2,10,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %9 = torch.vtensor.literal(dense_resource<torch_tensor_128_128_torch.float32_1> : tensor<128x128xf32>) : !torch.vtensor<[128,128],f32>
    %10 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32_2> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %11 = torch.aten.linear %5, %9, %10 : !torch.vtensor<[2,10,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %12 = torch.vtensor.literal(dense_resource<torch_tensor_128_128_torch.float32_2> : tensor<128x128xf32>) : !torch.vtensor<[128,128],f32>
    %13 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32_3> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %14 = torch.aten.linear %5, %12, %13 : !torch.vtensor<[2,10,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %int-2 = torch.constant.int -2
    %int-1_1 = torch.constant.int -1
    %15 = torch.aten.transpose.int %11, %int-2, %int-1_1 : !torch.vtensor<[2,10,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,128,10],f32>
    %16 = torch.aten.matmul %8, %15 : !torch.vtensor<[2,10,128],f32>, !torch.vtensor<[2,128,10],f32> -> !torch.vtensor<[2,10,10],f32>
    %float1.131370e01 = torch.constant.float 11.313708498984761
    %17 = torch.aten.div.Scalar %16, %float1.131370e01 : !torch.vtensor<[2,10,10],f32>, !torch.float -> !torch.vtensor<[2,10,10],f32>
    %int-1_2 = torch.constant.int -1
    %none = torch.constant.none
    %18 = torch.aten.softmax.int %17, %int-1_2, %none : !torch.vtensor<[2,10,10],f32>, !torch.int, !torch.none -> !torch.vtensor<[2,10,10],f32>
    %19 = torch.aten.matmul %18, %14 : !torch.vtensor<[2,10,10],f32>, !torch.vtensor<[2,10,128],f32> -> !torch.vtensor<[2,10,128],f32>
    return %19 : !torch.vtensor<[2,10,128],f32>
  }
}


