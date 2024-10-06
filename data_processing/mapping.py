import json
import sys

def process_json(input_path, output_path):
    # Load the input JSON file
    with open(input_path, 'r') as input_file:
        data = json.load(input_file)

    # Initialize dictionaries for audio and non-audio categories
    output_data = {"verbal": [], "non-verbal": []}

    # Process each item in the JSON
    for item in data:
        if item.get('subtitle', '').strip():
            output_data["verbal"].append(item["video_id"])
        else:
            output_data["non-verbal"].append(item["video_id"])

    # Write the output to a JSON file
    with open(output_path, 'w') as output_file:
        json.dump(output_data, output_file, indent=4)

if __name__ == "__main__":
    # Take input and output file paths from command line arguments
    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <input_path> <output_path>")
    #     sys.exit(1)

    input_path = '../data/test120/sample_v_nv_test120_new.json'
    output_path = '../data/test120/test120_mapping.json'

    # Process the JSON file
    process_json(input_path, output_path)