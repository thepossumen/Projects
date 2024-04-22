import os
import json

def update_json(query, answer, context, model, query_type, truth, json_path="system_eval.json", model_info_path="model_info.json"):
    data = {}
    # Check if the JSON file exists
    if os.path.exists(json_path):
        # If it exists, read the existing JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

    # Check if the query already exists
    if query not in data:
        # If not, create a new dictionary for this query
        data[query] = {"type": query_type, "truth": truth, "context": context}

    # Get the model ID from the model info
    model_id = update_model_info(model, model_info_path)

    # Update the answer for the given model version
    data[query][model_id] = answer

    # Save the updated JSON
    with open(json_path, 'w') as f:
        json.dump(data, f)

def update_model_info(model, model_info_path="model_info.json"):
    model_info = {}

    # Check if the JSON file exists
    if os.path.exists(model_info_path):
        # If it exists, read the existing JSON file
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)

    # Extract the specified parameters
    params_dict = vars(model.params)
    params = {key: params_dict[key] for key in ["system_input", "k", "exact_search", "MAX_NEW_TOKENS", "TEMP", "DO_SAMPLE", "NUM_BEAMS", "SKIP_SPECIAL_TOKENS", "ADD_SPECIAL_TOKENS"]}

    # Check if a model with the same name and specified parameters already exists
    for model_id, info in model_info.items():
        # Sort the dictionaries before comparing
        if info["name"] == model.model.name and sorted(info["parameters"].items()) == sorted(params.items()):
            # If it exists, return the corresponding model ID
            return model_id

    # If it doesn't exist, generate a new model ID
    new_model_id = f"model_{len(model_info) + 1}"

    # Add a new entry to the JSON with the new model ID, name, and specified parameters
    model_info[new_model_id] = {"name": model.model.name, "parameters": params}

    # Save the updated JSON
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f)

    # Return the new model ID
    return new_model_id


if __name__ == "__main__":
    # Define the path to the JSON files
    json_path = 'system_eval.json'
    model_info_path = 'model_info.json'

    # Define the parameters for a model version
    class GP():
        def __init__(self):
            self.temperature = 0.5
            self.random_sampling = True
            self.num_beams = 1
    parameters = GP()

    # Define a model
    class Model:
        def __init__(self, name, parameters):
            self.name = name
            self.params = parameters

    model_v1 = Model('model_robocop', parameters)
    model_v2 = Model('model_terminator', parameters)

    # Update the model info
    model_id = update_model_info(model_v1, model_info_path)
    model_id2 = update_model_info(model_v2, model_info_path)

    # Call the update_json function three times with different examples of query and answer
    update_json('What is the capital of France?', 'Paris', model_v1, 'general', 'Paris', json_path)
    update_json('What is the highest mountain in the world?', 'Mount Everest', model_v1, 'specific', 'Mount Everest', json_path)
    update_json('Who wrote "Pride and Prejudice"?', 'Jane Austen', model_v1, 'specific', 'Jane Austen', json_path)
    update_json('What is the capital of Sweden?', 'Stockholm', model_v1, 'general', 'Stockholm', json_path)
    update_json('Who wrote "To Kill a Mockingbird"?', 'Harper Lee', model_v1, 'specific', 'Harper Lee', json_path)

    # Call the update_json function with different examples of query and answer for the second model
    update_json('What is the capital of Sweden?', 'Stockholm', model_v2, 'general', 'Stockholm', json_path)
    update_json('What is the highest mountain in the world?', 'Mount Everest', model_v2, 'specific', 'Mount Everest', json_path)
    update_json('Who wrote "To Kill a Mockingbird"?', 'Harper Lee', model_v2, 'specific', 'Harper Lee', json_path)
    