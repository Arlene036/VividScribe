import json
import sys

def process_valor_json(input_path, non_verbal_path):
    # Load the input JSON file
    with open(input_path, 'r') as input_file:
        data = json.load(input_file)

    # Initialize dictionaries for audio and non-audio categories
    non_verbal = []

    # Process each item in the JSON
    for item in data:
        if not item.get('subtitle', '').strip():
            non_verbal.append(item)
    
    print(f"Number of non-verbal data: {len(non_verbal)}")
    
    # Write the non-verbal data to a JSON file
    with open(non_verbal_path, 'w') as non_verbal_file:
        json.dump(non_verbal, non_verbal_file, indent=4)


if __name__ == "__main__":
    # Take input and output file paths from command line arguments
    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <input_path> <output_path>")
    #     sys.exit(1)

    valor_input_path = 'data/valor120/sample_v_nv_test120_new.json'
    valor_non_verbal_path = 'data/valor120/valor60_non_verbal.json'

    # Process the JSON file
    process_valor_json(valor_input_path, valor_non_verbal_path)
    
