#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @main(%arg0: !torch.vtensor<[2,10],si64>) -> !torch.vtensor<[2,10,128],f32> {
    %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[2,10],si64> -> tensor<2x10xi64>
    %1 = torch.vtensor.literal(dense_resource<torch_tensor_1000_64_torch.float32> : tensor<1000x64xf32>) : !torch.vtensor<[1000,64],f32>
    %2 = torch_c.to_builtin_tensor %1 : !torch.vtensor<[1000,64],f32> -> tensor<1000x64xf32>
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %false_0 = torch.constant.bool false
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %3 = tensor.empty() : tensor<2x10x64xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0 : tensor<2x10xi64>) outs(%3 : tensor<2x10x64xf32>) {
    ^bb0(%in: i64, %out: f32):
      %45 = arith.index_cast %in : i64 to index
      %46 = linalg.index 2 : index
      %47 = arith.index_cast %in : i64 to index
      %c0_107 = arith.constant 0 : index
      %c1000 = arith.constant 1000 : index
      %48 = arith.cmpi slt, %47, %c1000 : index
      cf.assert %48, "index must be smaller than dim size"
      %c0_i64_108 = arith.constant 0 : i64
      %49 = arith.cmpi sge, %in, %c0_i64_108 : i64
      cf.assert %49, "index must be larger or equal to 0"
      %extracted = tensor.extract %2[%45, %46] : tensor<1000x64xf32>
      linalg.yield %extracted : f32
    } -> tensor<2x10x64xf32>
    %cast = tensor.cast %4 : tensor<2x10x64xf32> to tensor<2x10x64xf32>
    %5 = torch_c.from_builtin_tensor %cast : tensor<2x10x64xf32> -> !torch.vtensor<[2,10,64],f32>
    %6 = torch.vtensor.literal(dense_resource<torch_tensor_128_64_torch.float32> : tensor<128x64xf32>) : !torch.vtensor<[128,64],f32>
    %7 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %8 = torch.aten.linear %5, %6, %7 : !torch.vtensor<[2,10,64],f32>, !torch.vtensor<[128,64],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %9 = torch_c.to_builtin_tensor %8 : !torch.vtensor<[2,10,128],f32> -> tensor<2x10x128xf32>
    %str = torch.constant.str "none"
    %c1_2 = arith.constant 1 : index
    %c0_3 = arith.constant 0 : index
    %c2_4 = arith.constant 2 : index
    %c1_5 = arith.constant 1 : index
    %c10_6 = arith.constant 10 : index
    %c2_7 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %10 = tensor.empty() : tensor<2x10x128xf32>
    %11 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<2x10x128xf32>) outs(%10 : tensor<2x10x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_107 = arith.constant 0.000000e+00 : f32
      %cst_108 = arith.constant 1.000000e+00 : f32
      %45 = arith.subf %in, %cst_107 : f32
      %cst_109 = arith.constant 2.000000e+00 : f32
      %46 = math.sqrt %cst_109 : f32
      %47 = arith.divf %45, %46 : f32
      %48 = math.erf %47 : f32
      %cst_110 = arith.constant 1.000000e+00 : f32
      %49 = arith.addf %cst_110, %48 : f32
      %cst_111 = arith.constant 5.000000e-01 : f32
      %50 = arith.mulf %cst_111, %49 : f32
      %51 = arith.mulf %in, %50 : f32
      linalg.yield %51 : f32
    } -> tensor<2x10x128xf32>
    %cast_8 = tensor.cast %11 : tensor<2x10x128xf32> to tensor<2x10x128xf32>
    %12 = torch_c.from_builtin_tensor %cast_8 : tensor<2x10x128xf32> -> !torch.vtensor<[2,10,128],f32>
    %13 = torch.vtensor.literal(dense_resource<torch_tensor_128_128_torch.float32> : tensor<128x128xf32>) : !torch.vtensor<[128,128],f32>
    %14 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32_1> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %15 = torch.aten.linear %12, %13, %14 : !torch.vtensor<[2,10,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %16 = torch_c.to_builtin_tensor %15 : !torch.vtensor<[2,10,128],f32> -> tensor<2x10x128xf32>
    %17 = torch.vtensor.literal(dense_resource<torch_tensor_128_128_torch.float32_1> : tensor<128x128xf32>) : !torch.vtensor<[128,128],f32>
    %18 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32_2> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %19 = torch.aten.linear %12, %17, %18 : !torch.vtensor<[2,10,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %20 = torch_c.to_builtin_tensor %19 : !torch.vtensor<[2,10,128],f32> -> tensor<2x10x128xf32>
    %21 = torch.vtensor.literal(dense_resource<torch_tensor_128_128_torch.float32_2> : tensor<128x128xf32>) : !torch.vtensor<[128,128],f32>
    %22 = torch.vtensor.literal(dense_resource<torch_tensor_128_torch.float32_3> : tensor<128xf32>) : !torch.vtensor<[128],f32>
    %23 = torch.aten.linear %12, %21, %22 : !torch.vtensor<[2,10,128],f32>, !torch.vtensor<[128,128],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,10,128],f32>
    %24 = torch_c.to_builtin_tensor %23 : !torch.vtensor<[2,10,128],f32> -> tensor<2x10x128xf32>
    %int-2 = torch.constant.int -2
    %int-1_9 = torch.constant.int -1
    %c0_10 = arith.constant 0 : index
    %c2_11 = arith.constant 2 : index
    %c1_12 = arith.constant 1 : index
    %c10_13 = arith.constant 10 : index
    %c2_14 = arith.constant 2 : index
    %c128_15 = arith.constant 128 : index
    %25 = tensor.empty() : tensor<2x128x10xf32>
    %transposed = linalg.transpose ins(%20 : tensor<2x10x128xf32>) outs(%25 : tensor<2x128x10xf32>) permutation = [0, 2, 1] 
    %cast_16 = tensor.cast %transposed : tensor<2x128x10xf32> to tensor<2x128x10xf32>
    %c0_17 = arith.constant 0 : index
    %c2_18 = arith.constant 2 : index
    %c0_19 = arith.constant 0 : index
    %c2_20 = arith.constant 2 : index
    %c2_21 = arith.constant 2 : index
    %c1_22 = arith.constant 1 : index
    %c10_23 = arith.constant 10 : index
    %c2_24 = arith.constant 2 : index
    %c128_25 = arith.constant 128 : index
    %c1_26 = arith.constant 1 : index
    %c128_27 = arith.constant 128 : index
    %c2_28 = arith.constant 2 : index
    %c10_29 = arith.constant 10 : index
    %c128_i64 = arith.constant 128 : i64
    %c128_i64_30 = arith.constant 128 : i64
    %26 = arith.cmpi eq, %c128_i64, %c128_i64_30 : i64
    cf.assert %26, "mismatching contracting dimension"
    %c2_i64 = arith.constant 2 : i64
    %c2_i64_31 = arith.constant 2 : i64
    %c10_i64 = arith.constant 10 : i64
    %c128_i64_32 = arith.constant 128 : i64
    %c128_i64_33 = arith.constant 128 : i64
    %c10_i64_34 = arith.constant 10 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0_35 = arith.constant 0 : index
    %c1_36 = arith.constant 1 : index
    %c0_37 = arith.constant 0 : index
    %c2_38 = arith.constant 2 : index
    %c1_39 = arith.constant 1 : index
    %c10_40 = arith.constant 10 : index
    %c2_41 = arith.constant 2 : index
    %c128_42 = arith.constant 128 : index
    %27 = tensor.empty(%c2_38, %c10_40, %c128_42) : tensor<?x?x?xf32>
    %cast_43 = tensor.cast %16 : tensor<2x10x128xf32> to tensor<?x?x?xf32>
    %c0_i64_44 = arith.constant 0 : i64
    %c0_45 = arith.constant 0 : index
    %c1_46 = arith.constant 1 : index
    %c0_47 = arith.constant 0 : index
    %c2_48 = arith.constant 2 : index
    %c1_49 = arith.constant 1 : index
    %c128_50 = arith.constant 128 : index
    %c2_51 = arith.constant 2 : index
    %c10_52 = arith.constant 10 : index
    %28 = tensor.empty(%c2_48, %c128_50, %c10_52) : tensor<?x?x?xf32>
    %cast_53 = tensor.cast %cast_16 : tensor<2x128x10xf32> to tensor<?x?x?xf32>
    %29 = tensor.empty() : tensor<2x10x10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %30 = linalg.fill ins(%cst : f32) outs(%29 : tensor<2x10x10xf32>) -> tensor<2x10x10xf32>
    %31 = linalg.batch_matmul ins(%cast_43, %cast_53 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%30 : tensor<2x10x10xf32>) -> tensor<2x10x10xf32>
    %cast_54 = tensor.cast %31 : tensor<2x10x10xf32> to tensor<2x10x10xf32>
    %float1.131370e01 = torch.constant.float 11.313708498984761
    %32 = torch_c.to_f64 %float1.131370e01
    %c1_55 = arith.constant 1 : index
    %c0_56 = arith.constant 0 : index
    %c2_57 = arith.constant 2 : index
    %c1_58 = arith.constant 1 : index
    %c10_59 = arith.constant 10 : index
    %c2_60 = arith.constant 2 : index
    %c10_61 = arith.constant 10 : index
    %33 = tensor.empty() : tensor<2x10x10xf32>
    %34 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cast_54 : tensor<2x10x10xf32>) outs(%33 : tensor<2x10x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %45 = arith.truncf %32 : f64 to f32
      %46 = arith.divf %in, %45 : f32
      linalg.yield %46 : f32
    } -> tensor<2x10x10xf32>
    %cast_62 = tensor.cast %34 : tensor<2x10x10xf32> to tensor<2x10x10xf32>
    %35 = torch_c.from_builtin_tensor %cast_62 : tensor<2x10x10xf32> -> !torch.vtensor<[2,10,10],f32>
    %int-1_63 = torch.constant.int -1
    %none = torch.constant.none
    %36 = torch.aten.softmax.int %35, %int-1_63, %none : !torch.vtensor<[2,10,10],f32>, !torch.int, !torch.none -> !torch.vtensor<[2,10,10],f32>
    %37 = torch_c.to_builtin_tensor %36 : !torch.vtensor<[2,10,10],f32> -> tensor<2x10x10xf32>
    %c0_64 = arith.constant 0 : index
    %c2_65 = arith.constant 2 : index
    %c0_66 = arith.constant 0 : index
    %c2_67 = arith.constant 2 : index
    %c2_68 = arith.constant 2 : index
    %c1_69 = arith.constant 1 : index
    %c10_70 = arith.constant 10 : index
    %c2_71 = arith.constant 2 : index
    %c10_72 = arith.constant 10 : index
    %c1_73 = arith.constant 1 : index
    %c10_74 = arith.constant 10 : index
    %c2_75 = arith.constant 2 : index
    %c128_76 = arith.constant 128 : index
    %c10_i64_77 = arith.constant 10 : i64
    %c10_i64_78 = arith.constant 10 : i64
    %38 = arith.cmpi eq, %c10_i64_77, %c10_i64_78 : i64
    cf.assert %38, "mismatching contracting dimension"
    %c2_i64_79 = arith.constant 2 : i64
    %c2_i64_80 = arith.constant 2 : i64
    %c10_i64_81 = arith.constant 10 : i64
    %c10_i64_82 = arith.constant 10 : i64
    %c10_i64_83 = arith.constant 10 : i64
    %c128_i64_84 = arith.constant 128 : i64
    %c0_i64_85 = arith.constant 0 : i64
    %c0_86 = arith.constant 0 : index
    %c1_87 = arith.constant 1 : index
    %c0_88 = arith.constant 0 : index
    %c2_89 = arith.constant 2 : index
    %c1_90 = arith.constant 1 : index
    %c10_91 = arith.constant 10 : index
    %c2_92 = arith.constant 2 : index
    %c10_93 = arith.constant 10 : index
    %39 = tensor.empty(%c2_89, %c10_91, %c10_93) : tensor<?x?x?xf32>
    %cast_94 = tensor.cast %37 : tensor<2x10x10xf32> to tensor<?x?x?xf32>
    %c0_i64_95 = arith.constant 0 : i64
    %c0_96 = arith.constant 0 : index
    %c1_97 = arith.constant 1 : index
    %c0_98 = arith.constant 0 : index
    %c2_99 = arith.constant 2 : index
    %c1_100 = arith.constant 1 : index
    %c10_101 = arith.constant 10 : index
    %c2_102 = arith.constant 2 : index
    %c128_103 = arith.constant 128 : index
    %40 = tensor.empty(%c2_99, %c10_101, %c128_103) : tensor<?x?x?xf32>
    %cast_104 = tensor.cast %24 : tensor<2x10x128xf32> to tensor<?x?x?xf32>
    %41 = tensor.empty() : tensor<2x10x128xf32>
    %cst_105 = arith.constant 0.000000e+00 : f32
    %42 = linalg.fill ins(%cst_105 : f32) outs(%41 : tensor<2x10x128xf32>) -> tensor<2x10x128xf32>
    %43 = linalg.batch_matmul ins(%cast_94, %cast_104 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%42 : tensor<2x10x128xf32>) -> tensor<2x10x128xf32>
    %cast_106 = tensor.cast %43 : tensor<2x10x128xf32> to tensor<2x10x128xf32>
    %44 = torch_c.from_builtin_tensor %cast_106 : tensor<2x10x128xf32> -> !torch.vtensor<[2,10,128],f32>
    return %44 : !torch.vtensor<[2,10,128],f32>
  }
}

