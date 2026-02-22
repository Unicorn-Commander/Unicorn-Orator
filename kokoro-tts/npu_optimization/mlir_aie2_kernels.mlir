// MLIR-AIE2 Kernels for WhisperX NPU Acceleration
// ==============================================
// Custom kernels for AMD NPU Phoenix (Ryzen AI)
// Target: AIE2 architecture with 1024-bit vector units

module @whisperx_npu_kernels {
  // Constants for AIE2 architecture
  memref.global "private" constant @VECTOR_WIDTH : i32 = 32  // 32 x int8 = 256 bits per lane
  memref.global "private" constant @AIE_TILES : i32 = 20     // 20 AIE tiles on Phoenix
  memref.global "private" constant @DMA_CHANNELS : i32 = 2   // 2 DMA channels per tile

  // Whisper Attention Kernel - Optimized for AIE2
  // This is the most compute-intensive part of Whisper
  aie.device(npu1_4col) {
    // Define AIE tile array layout (4x5 array)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    
    // Memory layout for efficient DMA
    %mem_a = aie.buffer(%tile_0_0) {sym_name = "query_buffer"} : memref<1024xi8>
    %mem_b = aie.buffer(%tile_0_1) {sym_name = "key_buffer"} : memref<1024xi8>
    %mem_c = aie.buffer(%tile_0_2) {sym_name = "value_buffer"} : memref<1024xi8>
    %mem_out = aie.buffer(%tile_0_3) {sym_name = "output_buffer"} : memref<1024xi8>

    // Attention Score Computation Kernel
    aie.core(%tile_0_0) {
      // Q @ K^T computation with INT8
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      
      scf.for %i = %c0 to %c1024 step %c32 {
        // Load 32 INT8 values (256 bits) in one instruction
        %q_vec = aie.load_vector %mem_a[%i] : memref<1024xi8>, vector<32xi8>
        %k_vec = aie.load_vector %mem_b[%i] : memref<1024xi8>, vector<32xi8>
        
        // AIE2 native MAC operation - 32 INT8 ops in parallel
        %acc = aie.mac %q_vec, %k_vec : vector<32xi8>, vector<32xi8> -> vector<32xi32>
        
        // Quantization-aware scaling
        %scale = arith.constant 127 : i32  // Q7 scaling factor
        %scaled = aie.mul %acc, %scale : vector<32xi32>
        
        // Store intermediate results
        aie.store_vector %scaled, %mem_out[%i] : vector<32xi32>, memref<1024xi8>
      }
      aie.end
    }

    // Softmax Kernel - Optimized for INT8
    aie.core(%tile_0_1) {
      // Efficient INT8 softmax using lookup tables
      %lut = aie.buffer(%tile_0_1) {sym_name = "exp_lut"} : memref<256xi8>
      
      // Initialize exponential lookup table
      affine.for %i = 0 to 256 {
        %idx = arith.index_cast %i : index to i8
        %exp_val = math.exp %idx : i8  // Approximated exp for INT8
        memref.store %exp_val, %lut[%i] : memref<256xi8>
      }
      
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      
      // Vectorized softmax
      scf.for %i = %c0 to %c1024 step %c32 {
        %scores = aie.load_vector %mem_out[%i] : memref<1024xi8>, vector<32xi8>
        
        // Lookup-based exponential
        %exp_scores = aie.lookup %scores, %lut : vector<32xi8>, memref<256xi8> -> vector<32xi8>
        
        // Parallel reduction for sum
        %sum = aie.reduce_add %exp_scores : vector<32xi8> -> i32
        
        // Normalize (division approximated with multiplication)
        %norm_factor = aie.reciprocal %sum : i32 -> i8
        %normalized = aie.mul %exp_scores, %norm_factor : vector<32xi8>, i8 -> vector<32xi8>
        
        aie.store_vector %normalized, %mem_out[%i] : vector<32xi8>, memref<1024xi8>
      }
      aie.end
    }

    // Matrix Multiply for Attention @ Values
    aie.core(%tile_0_2) {
      // Tiled matrix multiplication for V projection
      %M = arith.constant 64 : index   // Sequence length (tiled)
      %K = arith.constant 64 : index   // Hidden dim (tiled)
      %N = arith.constant 64 : index   // Head dim
      
      // Triple nested loop with AIE2 optimizations
      affine.for %m = 0 to %M step 8 {           // 8x unroll for output
        affine.for %n = 0 to %N step 8 {         // 8x unroll for output  
          // Initialize 8x8 accumulator tile
          %acc_tile = aie.zero : vector<64xi32>
          
          affine.for %k = 0 to %K step 32 {      // 32x vector width
            // Load attention scores and values
            %att_vec = aie.load_vector %mem_out[%m * %K + %k] : memref<1024xi8>, vector<32xi8>
            %val_vec = aie.load_vector %mem_c[%k * %N + %n] : memref<1024xi8>, vector<32xi8>
            
            // Outer product accumulation
            %prod = aie.mac_outer %att_vec, %val_vec, %acc_tile : 
                    vector<32xi8>, vector<32xi8>, vector<64xi32> -> vector<64xi32>
            %acc_tile = %prod
          }
          
          // Quantize back to INT8
          %quant_tile = aie.quantize %acc_tile : vector<64xi32> -> vector<64xi8>
          
          // Store 8x8 output tile
          aie.store_tile %quant_tile, %mem_out[%m * %N + %n] : vector<64xi8>, memref<1024xi8>
        }
      }
      aie.end
    }

    // DMA Configuration for Streaming
    aie.flow(%tile_0_0, "DMA0" : 0, %tile_0_1, "DMA0" : 0)
    aie.flow(%tile_0_1, "DMA0" : 0, %tile_0_2, "DMA0" : 0)
    aie.flow(%tile_0_2, "DMA0" : 0, %tile_0_3, "DMA0" : 0)
  }

  // Mel Spectrogram Feature Extraction Kernel
  func.func @mel_spectrogram_aie2(%audio: memref<16000xi16>, %mel_out: memref<80x3000xi8>) {
    // FFT window parameters
    %window_size = arith.constant 400 : index    // 25ms @ 16kHz
    %hop_size = arith.constant 160 : index       // 10ms @ 16kHz
    %n_mels = arith.constant 80 : index
    
    // Process audio in chunks suitable for AIE2
    affine.for %frame = 0 to 3000 {
      %offset = arith.muli %frame, %hop_size : index
      
      // Window and FFT preparation
      %windowed = memref.alloca() : memref<512xi16>  // Zero-padded to 512
      
      // Apply Hanning window with AIE2 vectors
      affine.for %i = 0 to %window_size step 32 {
        %audio_vec = vector.load %audio[%offset + %i] : memref<16000xi16>, vector<32xi16>
        %window_vec = aie.hanning_window %i : vector<32xi16>
        %windowed_vec = arith.muli %audio_vec, %window_vec : vector<32xi16>
        vector.store %windowed_vec, %windowed[%i] : memref<512xi16>, vector<32xi16>
      }
      
      // FFT using AIE2 butterfly operations
      %fft_real = memref.alloca() : memref<256xi32>
      %fft_imag = memref.alloca() : memref<256xi32>
      
      // Radix-4 FFT optimized for AIE2
      call @radix4_fft_aie2(%windowed, %fft_real, %fft_imag) : 
           (memref<512xi16>, memref<256xi32>, memref<256xi32>) -> ()
      
      // Mel filterbank application
      affine.for %mel = 0 to %n_mels {
        %energy = arith.constant 0 : i32
        
        // Apply triangular mel filter
        %start_bin = call @mel_filter_start(%mel) : (index) -> index
        %end_bin = call @mel_filter_end(%mel) : (index) -> index
        
        affine.for %bin = %start_bin to %end_bin {
          %real = memref.load %fft_real[%bin] : memref<256xi32>
          %imag = memref.load %fft_imag[%bin] : memref<256xi32>
          
          // Magnitude squared
          %mag2 = arith.muli %real, %real : i32
          %mag2_i = arith.muli %imag, %imag : i32
          %power = arith.addi %mag2, %mag2_i : i32
          
          // Apply mel weight
          %weight = call @mel_filter_weight(%mel, %bin) : (index, index) -> i32
          %weighted = arith.muli %power, %weight : i32
          %energy = arith.addi %energy, %weighted : i32
        }
        
        // Log mel energy with INT8 quantization
        %log_energy = call @fast_log_aie2(%energy) : (i32) -> i8
        memref.store %log_energy, %mel_out[%mel, %frame] : memref<80x3000xi8>
      }
    }
    return
  }

  // Radix-4 FFT kernel for AIE2
  func.func private @radix4_fft_aie2(%in: memref<512xi16>, 
                                     %out_real: memref<256xi32>, 
                                     %out_imag: memref<256xi32>) {
    // Bit-reversal using AIE2 shuffle instructions
    %shuffled = memref.alloca() : memref<512xi16>
    
    affine.for %i = 0 to 512 step 32 {
      %vec = vector.load %in[%i] : memref<512xi16>, vector<32xi16>
      %rev_indices = aie.bit_reverse_indices %i : vector<32xi32>
      %shuffled_vec = aie.shuffle %vec, %rev_indices : vector<32xi16>, vector<32xi32> -> vector<32xi16>
      vector.store %shuffled_vec, %shuffled[%i] : memref<512xi16>, vector<32xi16>
    }
    
    // Radix-4 butterfly stages
    %stages = arith.constant 5 : index  // log4(256) = 4, but we need 5 for 512->256
    
    affine.for %stage = 0 to %stages {
      %stride = arith.shli %c1, %stage : index
      %groups = arith.shrui %c256, %stride : index
      
      affine.for %group = 0 to %groups {
        affine.for %pair = 0 to %stride {
          // Radix-4 butterfly with twiddle factors
          %idx0 = arith.addi %group, %pair : index
          %idx1 = arith.addi %idx0, %stride : index
          %idx2 = arith.addi %idx1, %stride : index
          %idx3 = arith.addi %idx2, %stride : index
          
          // Load 4 complex values
          %a = vector.load %shuffled[%idx0] : memref<512xi16>, vector<2xi16>
          %b = vector.load %shuffled[%idx1] : memref<512xi16>, vector<2xi16>
          %c = vector.load %shuffled[%idx2] : memref<512xi16>, vector<2xi16>
          %d = vector.load %shuffled[%idx3] : memref<512xi16>, vector<2xi16>
          
          // Get twiddle factors from LUT
          %tw1 = call @twiddle_factor(%stage, %pair, 1) : (index, index, index) -> vector<2xi16>
          %tw2 = call @twiddle_factor(%stage, %pair, 2) : (index, index, index) -> vector<2xi16>
          %tw3 = call @twiddle_factor(%stage, %pair, 3) : (index, index, index) -> vector<2xi16>
          
          // Complex multiplication with twiddles
          %b_tw = aie.complex_mul %b, %tw1 : vector<2xi16>
          %c_tw = aie.complex_mul %c, %tw2 : vector<2xi16>
          %d_tw = aie.complex_mul %d, %tw3 : vector<2xi16>
          
          // Radix-4 butterfly
          %sum0 = arith.addi %a, %b_tw : vector<2xi16>
          %sum1 = arith.addi %c_tw, %d_tw : vector<2xi16>
          %diff0 = arith.subi %a, %b_tw : vector<2xi16>
          %diff1 = arith.subi %c_tw, %d_tw : vector<2xi16>
          
          %out0 = arith.addi %sum0, %sum1 : vector<2xi16>
          %out1 = arith.subi %sum0, %sum1 : vector<2xi16>
          %out2 = arith.addi %diff0, %diff1 : vector<2xi16>
          %out3 = arith.subi %diff0, %diff1 : vector<2xi16>
          
          // Store back
          vector.store %out0, %shuffled[%idx0] : memref<512xi16>, vector<2xi16>
          vector.store %out1, %shuffled[%idx1] : memref<512xi16>, vector<2xi16>
          vector.store %out2, %shuffled[%idx2] : memref<512xi16>, vector<2xi16>
          vector.store %out3, %shuffled[%idx3] : memref<512xi16>, vector<2xi16>
        }
      }
    }
    
    // Extract real and imaginary parts for first 256 bins
    affine.for %i = 0 to 256 {
      %complex = vector.load %shuffled[%i * 2] : memref<512xi16>, vector<2xi16>
      %real = vector.extract %complex[0] : vector<2xi16>
      %imag = vector.extract %complex[1] : vector<2xi16>
      
      %real_i32 = arith.extsi %real : i16 to i32
      %imag_i32 = arith.extsi %imag : i16 to i32
      
      memref.store %real_i32, %out_real[%i] : memref<256xi32>
      memref.store %imag_i32, %out_imag[%i] : memref<256xi32>
    }
    
    return
  }

  // Fast logarithm approximation for AIE2
  func.func private @fast_log_aie2(%x: i32) -> i8 {
    // Count leading zeros for fast log2
    %clz = aie.clz %x : i32
    %log2 = arith.subi %c31, %clz : i32
    
    // Linear interpolation for fractional part
    %shift = arith.subi %c31, %log2 : i32
    %normalized = arith.shli %x, %shift : i32
    
    // Lookup table for log correction
    %lut_idx = arith.shrui %normalized, %c24 : i32  // Top 8 bits
    %correction = aie.lookup %lut_idx, @log_correction_lut : i32, memref<256xi8> -> i8
    
    // Combine integer and fractional parts
    %log2_i8 = arith.trunci %log2 : i32 to i8
    %result = arith.addi %log2_i8, %correction : i8
    
    return %result : i8
  }

  // Convolution kernel for encoder layers
  func.func @conv1d_aie2(%input: memref<3000x512xi8>, 
                         %weight: memref<3x512x512xi8>,
                         %output: memref<3000x512xi8>) {
    %M = arith.constant 3000 : index  // Time dimension
    %C_in = arith.constant 512 : index
    %C_out = arith.constant 512 : index
    %K = arith.constant 3 : index      // Kernel size
    
    // Process multiple output channels in parallel
    affine.for %oc = 0 to %C_out step 8 {      // 8 output channels at once
      affine.for %t = 1 to %M - 1 {            // Handle padding
        // Initialize 8 accumulators
        %acc = aie.zero : vector<8xi32>
        
        // Convolution kernel
        affine.for %k = 0 to %K {
          %t_in = arith.addi %t, %k : index
          %t_in = arith.subi %t_in, %c1 : index  // Center kernel
          
          affine.for %ic = 0 to %C_in step 32 {  // Process 32 input channels
            // Load input vector
            %in_vec = vector.load %input[%t_in, %ic] : memref<3000x512xi8>, vector<32xi8>
            
            // Load 8x32 weight tile
            affine.for %oc_off = 0 to 8 {
              %w_vec = vector.load %weight[%k, %ic, %oc + %oc_off] : memref<3x512x512xi8>, vector<32xi8>
              
              // MAC operation
              %dot = aie.dot %in_vec, %w_vec : vector<32xi8>, vector<32xi8> -> i32
              %acc_elem = vector.extract %acc[%oc_off] : vector<8xi32>
              %new_elem = arith.addi %acc_elem, %dot : i32
              %acc = vector.insert %new_elem, %acc[%oc_off] : i32 into vector<8xi32>
            }
          }
        }
        
        // Quantize and store results
        %out_vec = aie.quantize %acc : vector<8xi32> -> vector<8xi8>
        vector.store %out_vec, %output[%t, %oc] : memref<3000x512xi8>, vector<8xi8>
      }
    }
    
    return
  }

  // Positional encoding addition
  func.func @add_positional_encoding_aie2(%input: memref<3000x512xi8>, 
                                          %pos_enc: memref<3000x512xi8>) {
    %M = arith.constant 3000 : index
    %D = arith.constant 512 : index
    
    // Vectorized addition with saturation
    affine.for %t = 0 to %M {
      affine.for %d = 0 to %D step 32 {
        %in_vec = vector.load %input[%t, %d] : memref<3000x512xi8>, vector<32xi8>
        %pos_vec = vector.load %pos_enc[%t, %d] : memref<3000x512xi8>, vector<32xi8>
        
        // Saturating addition for INT8
        %sum = aie.sadd %in_vec, %pos_vec : vector<32xi8>
        
        vector.store %sum, %input[%t, %d] : memref<3000x512xi8>, vector<32xi8>
      }
    }
    
    return
  }

  // Layer normalization for INT8
  func.func @layer_norm_aie2(%input: memref<512xi8>, %gamma: memref<512xi8>, 
                             %beta: memref<512xi8>, %output: memref<512xi8>) {
    %D = arith.constant 512 : index
    %c0 = arith.constant 0 : i32
    
    // Compute mean with tree reduction
    %sum = affine.for %d = 0 to %D step 32 iter_args(%acc = %c0) -> i32 {
      %vec = vector.load %input[%d] : memref<512xi8>, vector<32xi8>
      %vec_i32 = arith.extsi %vec : vector<32xi8> to vector<32xi32>
      %partial_sum = aie.reduce_add %vec_i32 : vector<32xi32> -> i32
      %new_acc = arith.addi %acc, %partial_sum : i32
      affine.yield %new_acc : i32
    }
    
    // Compute mean (with rounding)
    %mean = arith.divsi %sum, %c512 : i32
    %mean_i8 = arith.trunci %mean : i32 to i8
    
    // Compute variance
    %var_acc = affine.for %d = 0 to %D step 32 iter_args(%acc = %c0) -> i32 {
      %vec = vector.load %input[%d] : memref<512xi8>, vector<32xi8>
      %mean_vec = vector.splat %mean_i8 : vector<32xi8>
      %diff = arith.subi %vec, %mean_vec : vector<32xi8>
      %diff_i16 = arith.extsi %diff : vector<32xi8> to vector<32xi16>
      %squared = arith.muli %diff_i16, %diff_i16 : vector<32xi16>
      %partial_var = aie.reduce_add %squared : vector<32xi16> -> i32
      %new_acc = arith.addi %acc, %partial_var : i32
      affine.yield %new_acc : i32
    }
    
    %var = arith.divsi %var_acc, %c512 : i32
    
    // Fast inverse square root approximation
    %inv_std = call @fast_rsqrt_aie2(%var) : (i32) -> i8
    
    // Normalize and apply affine transform
    affine.for %d = 0 to %D step 32 {
      %x_vec = vector.load %input[%d] : memref<512xi8>, vector<32xi8>
      %gamma_vec = vector.load %gamma[%d] : memref<512xi8>, vector<32xi8>
      %beta_vec = vector.load %beta[%d] : memref<512xi8>, vector<32xi8>
      
      %mean_vec = vector.splat %mean_i8 : vector<32xi8>
      %centered = arith.subi %x_vec, %mean_vec : vector<32xi8>
      
      %inv_std_vec = vector.splat %inv_std : vector<32xi8>
      %normalized = aie.mul %centered, %inv_std_vec : vector<32xi8>
      
      %scaled = aie.mul %normalized, %gamma_vec : vector<32xi8>
      %shifted = aie.sadd %scaled, %beta_vec : vector<32xi8>
      
      vector.store %shifted, %output[%d] : memref<512xi8>, vector<32xi8>
    }
    
    return
  }

  // Fast reciprocal square root for AIE2
  func.func private @fast_rsqrt_aie2(%x: i32) -> i8 {
    // Newton-Raphson approximation adapted for INT8
    %x_half = arith.shrui %x, %c1 : i32
    
    // Initial guess using magic constant
    %magic = arith.constant 0x5f3759df : i32
    %i = arith.subi %magic, %x_half : i32
    
    // One iteration of Newton-Raphson
    %y = arith.trunci %i : i32 to i8
    %y_sq = arith.muli %y, %y : i8
    %three = arith.constant 3 : i8
    %prod = arith.muli %x_half, %y_sq : i32
    %prod_i8 = arith.trunci %prod : i32 to i8
    %three_minus = arith.subi %three, %prod_i8 : i8
    %result = arith.muli %y, %three_minus : i8
    %result = arith.shrui %result, %c1 : i8  // Divide by 2
    
    return %result : i8
  }
}

// Helper functions for mel filterbank
func.func private @mel_filter_start(%mel: index) -> index
func.func private @mel_filter_end(%mel: index) -> index  
func.func private @mel_filter_weight(%mel: index, %bin: index) -> i32
func.func private @twiddle_factor(%stage: index, %idx: index, %k: index) -> vector<2xi16>

// Global lookup tables
memref.global "private" constant @log_correction_lut : memref<256xi8> = dense<[...]>
memref.global "private" constant @exp_lut_int8 : memref<256xi8> = dense<[...]>