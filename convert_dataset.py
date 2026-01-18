import pandas as pd
import os

def convert_qa_to_text():
    files = ['Training Data.csv', 'Testing Data.csv']
    combined_knowledge = []

    for file in files:
        if os.path.exists(file):
            # Load CSV, skip the header row, and don't use column names
            # header=0 means we use the first row as headers, but then we call them by index
            df = pd.read_csv(file)
            
            # This forces the code to look at column 1 and column 2 regardless of names
            for i in range(len(df)):
                instr = str(df.iloc[i, 0]) # First column
                out = str(df.iloc[i, 1])   # Second column
                
                if instr.lower() != 'nan' and out.lower() != 'nan':
                    qa_pair = f"Scenario: {instr}\nAdvice: {out}"
                    combined_knowledge.append(qa_pair)
    
    if not os.path.exists('data'):
        os.makedirs('data')

    with open(os.path.join('data', 'master_knowledge.txt'), 'w', encoding='utf-8') as f:
        f.write("\n---\n".join(combined_knowledge))
    
    print(f"SUCCESS! Processed {len(combined_knowledge)} entries using index-based loading.")

if __name__ == "__main__":
    convert_qa_to_text()