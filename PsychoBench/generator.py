from openai import OpenAI
import anthropic
import os
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import time
from generate_conversations import generate_response as generate_conv_response
from tqdm import tqdm


def convert_results(result, column_header):
    result = result.strip()  # Remove leading and trailing whitespace
    try:
        result_list = [int(element.strip()[-1]) for element in result.split('\n') if element.strip()]
    except:
        result_list = ["" for element in result.split('\n')]
        print(f"Unable to capture the responses on {column_header}.")
        
    return result_list

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def generate_response(model, inputs):
    """
    PsychoBench appelle generate_response(model, inputs),
    mais la vraie fonction attend generate_response(messages, model)
    → on adapte ici les arguments dans le bon ordre.
    """
    return generate_conv_response(messages=inputs, model=model)

def generator(questionnaire, args):
    testing_file = args.testing_file
    model = args.model
    records_file = args.name_exp if args.name_exp is not None else model
    conversation_history = f"{args.prompt_context}\n"
    # Read the existing CSV file into a pandas DataFrame
    df = pd.read_csv(testing_file)

    # Find the columns whose headers start with "order"
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    with tqdm(total=total_iterations) as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                # Find the index of the previous column
                questions_column_index = i - 1
                shuffle_count += 1
                
                # Retrieve the column data as a string
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+30] for i in range(0, len(questions_list), 30)]  
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)]) for j, questions in enumerate(separated_questions)]


                for k in range(args.test_count):
                    
                    df = pd.read_csv(testing_file)
                    
                    # Insert the updated column into the DataFrame with a unique identifier in the header
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'
                    
                    while(True):
                        result_string_list = []
                        previous_records = [{"role": "user", "content": conversation_history}]
                        
                        for questions_string in questions_list:
                            result = ''
                            if model == 'text-davinci-003':
                                inputs = questionnaire["inner_setting"].replace('Format: \"index: score\"', 'Format: \"index: score\\\n\"') + questionnaire["prompt"] + '\n' + questions_string
                                result = generate_response(model, inputs)
                            elif model.startswith("gpt") or model.startswith("claude"):
                                inputs = previous_records + [
                                    {"role": "user", "content": questionnaire["inner_setting"]},
                                    {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                                ]
                                result = generate_response(model, inputs)
                                previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                                previous_records.append({"role": "assistant", "content": result})
                            
                            else:
                                raise ValueError("The model is not supported or does not exist.")
                        
                            result_string_list.append(result.strip())
                        
                            # Write the prompts and results to the file
                            os.makedirs("prompts", exist_ok=True)
                            os.makedirs("responses", exist_ok=True)

                            with open(f'prompts/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{inputs}\n====\n')
                            with open(f'responses/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)
                        
                        result_list = convert_results(result_string, column_header)
                        
                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break
                        except:
                            print(f"Unable to capture the responses on {column_header}.")

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)
                    
                    pbar.update(1)
