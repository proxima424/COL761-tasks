To run the code :

$ make

$ bash compile.sh

for compression:

$ bash interface.sh C input_file output_compressed_file

for decompression:

$ bash interface.sh D output_compressed_file decompressed_file

for verification of lossless decompression: 

$ bash interface.sh V input_file decompressed_file

For running on HPC run following first:

$ module load compiler/gcc/9.1.0
