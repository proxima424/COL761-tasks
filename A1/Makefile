CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3

TARGET = main
SRCS = main.cpp fptree.cpp
OBJS = $(SRCS:.cpp=.o)

DECOMPRESS_TARGET = decompress
DECOMPRESS_SRCS = decompress.cpp
DECOMPRESS_OBJS = $(DECOMPRESS_SRCS:.cpp=.o)

VERIFY_TARGET = verify
VERIFY_SRCS = verify.cpp
VERIFY_OBJS = $(VERIFY_SRCS:.cpp=.o)

.PHONY: clean

all: $(TARGET) $(DECOMPRESS_TARGET) $(VERIFY_TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(DECOMPRESS_TARGET): $(DECOMPRESS_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(VERIFY_TARGET): $(VERIFY_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) ./test.dat ./compressed.dat

run_decompress: $(DECOMPRESS_TARGET)
	./$(DECOMPRESS_TARGET) ./compressed.dat ./decompressed.dat

run_verify: $(VERIFY_TARGET)
	./$(VERIFY_TARGET) ./decompressed.dat ./test.dat

clean:
	rm -f $(OBJS) $(DECOMPRESS_OBJS) $(VERIFY_OBJS) $(TARGET) $(DECOMPRESS_TARGET) $(VERIFY_TARGET)



