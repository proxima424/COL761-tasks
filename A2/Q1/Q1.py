import sys
import os
import re
import matplotlib.pyplot as plt 
import time
import subprocess
num_vertices=0
num_graphs=0
def get_numbers(line):
    # Use regular expression to extract numbers from the line
    numbers = re.findall(r'\d+', line)
    return numbers
def alphabet_to_numeric(alphabet):
    # Convert the alphabet to uppercase to handle both lowercase and uppercase characters
    alphabet = alphabet.upper()
    
    # Get the Unicode code point of the alphabet
    numeric_value = ord(alphabet) - 64  # Assuming A=1, B=2, ..., Z=26
    
    return numeric_value
def process_line_fsg(line):
    global num_vertices
    length=len(get_numbers(line))
    if line.startswith('#'):
        num_vertices=0
        return 't'+' '+'#' +' '+line.rstrip('\n')[1:]
    elif line[0].isalpha():
        vertices_str = ''.join(str(num_vertices))
        p='v'+' '+vertices_str+' '+str(alphabet_to_numeric(line[0]))
        num_vertices+=1
        return p
    elif (length>1):
        line=line.strip()
        return 'u'+' '+line

def process_line_others(line):
    global num_vertices
    global num_graphs
    length=len(get_numbers(line))
    if line.startswith('#'):
        num_vertices=0
        num_graphs+=1
        return 't'+ ' '+'#' +' '+ line.rstrip('\n')[1:]
    elif line[0].isalpha():
        vertices_str = ''.join(str(num_vertices))
        p='v'+' '+vertices_str+' '+str(alphabet_to_numeric(line[0]))
        num_vertices+=1
        return p
    elif (length>1):
        line=line.strip()
        return 'e'+' '+line
    
def create_input(input_file_path, output_file_path,algo):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            if (algo=='fsg'):
                processed_line=process_line_fsg(line)
            else:
                processed_line=process_line_others(line)
            if (processed_line):
                output_file.write(processed_line+'\n')
            

if __name__ == "__main__":
    path_to_data = sys.argv[1]
    path_to_result = sys.argv[2]

    ## Function to change the format of the input file and save it in path_to_result
    ##Create two files for u and e , that is for fsg and rest
    create_input(path_to_data,'input_fsg.txt','fsg')
    print("FSG DONE")
    create_input(path_to_data,'input_other.txt','gaston')
    print("Others DONE")
    ## Function to run each algorithm and plot
    supports = [90, 95,100]
    gspan = []
    fsg = []
    gaston = []
    present_directory = os.getcwd()

    for support in supports:
        
        t = time.time()
        os.system(f"timeout 1h {present_directory}/gSpan-64 -s {support/100} -f {'input_other.txt'} -o -i")
        gspan.append((time.time()-t)/60)
        t = time.time()
        os.system(f"timeout 1h {present_directory}/pafi-1.0.1/fsg -s {support} {'input_fsg.txt'}")
        fsg.append((time.time()-t)/60)
        t = time.time()
        os.system(f"timeout 1h {present_directory}/gaston-1.1/gaston {(num_graphs*support)/100} {'input_other.txt'} ")
        gaston.append((time.time()-t)/60)
    print('Subgraphs process completed ')



    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(supports, fsg, marker='o', label='fsg')
    plt.plot(supports, gspan, marker='o', label='gspan')
    plt.plot(supports, gaston, marker='o', label='gaston')
    plt.xlabel('Support Values')
    plt.ylabel('Running Time (seconds)')
    plt.title('Running Time of Algorithms for Different Support Values')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image or display it
    plt.savefig(path_to_result)
    # plt.show()  # Uncomment to display the plot interactively

    # Close the plot
    plt.close()
    #plt.savefig(path_to_result)
