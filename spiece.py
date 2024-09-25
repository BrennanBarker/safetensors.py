from google.protobuf import json_format
from sentencepiece import sentencepiece_model_pb2

def model_to_json(model_path, json_output_path):
    # Load the SentencePiece model
    model = sentencepiece_model_pb2.ModelProto()
    with open(model_path, 'rb') as f:
        model.ParseFromString(f.read())

    # Convert Protobuf model to JSON
    json_str = json_format.MessageToJson(model, preserving_proto_field_name=True)

    # Write the JSON string to a file
    with open(json_output_path, 'w') as json_file:
        json_file.write(json_str)

    print(f"Model converted to JSON and saved at {json_output_path}")

def json_to_model(json_input_path, model_output_path):
    # Load JSON data from file
    with open(json_input_path, 'r') as json_file:
        json_str = json_file.read()

    # Parse the JSON string into a Protobuf message
    model = sentencepiece_model_pb2.ModelProto()
    json_format.Parse(json_str, model)

    # Serialize the Protobuf message back into a .model file
    with open(model_output_path, 'wb') as model_file:
        model_file.write(model.SerializeToString())

    print(f"JSON converted back to model and saved at {model_output_path}")


if __name__ == '__main__':
    import filecmp

    example_file = 'spiece.model'
    intermediate_json = 'output.json'
    reconstructed_file = 'restored_example.model'

    model_to_json(example_file, intermediate_json)
    json_to_model(intermediate_json, reconstructed_file)

    cmp = filecmp.cmp(example_file, reconstructed_file, shallow=False)

    if cmp:
        print('reconstructed file is same as example')
    else:
        print('reconstructed file different from example')