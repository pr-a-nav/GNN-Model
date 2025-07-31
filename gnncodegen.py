import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import re
import os
from datasets import load_dataset
import pandas as pd
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch_geometric.nn import GATConv


# Initialize tokenizer and model for function name embeddings
tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
embedding_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

PY_LANGUAGE = Language(tspython.language())
parser = Parser()
parser.language=PY_LANGUAGE

node_types = ['module', 'function_definition', 'identifier', 'if_statement', 'return_statement', 'binary_expression', 'call']
node_type_dict = {t: i for i, t in enumerate(node_types)}

def extract_ast_nodes(node, nodes=None, edges=None, code_text=None):
    if nodes is None:
        nodes = []
    if edges is None:
        edges = []
    node_id = len(nodes)
    node_info = {"id": node_id, "type": node.type, "text": node.text.decode()}

    if node.type == 'function_definition':
        function_name_node = node.child_by_field_name('name')
        if function_name_node:
            node_info['function_name'] = function_name_node.text.decode()
    elif node.type == 'call':
         function_name_node = node.child_by_field_name('function')
         if function_name_node:
             node_info['function_name'] = function_name_node.text.decode()


    nodes.append(node_info)

    for child in node.children:
        child_id = len(nodes)
        edges.append((node_id, child_id))
        extract_ast_nodes(child, nodes, edges, code_text)
    return nodes, edges

# Create PyG Data object with node features
def create_graph(nodes, edges, label):
    x_type = torch.zeros(len(nodes), len(node_types))
    for node in nodes:
        node_type_idx = node_type_dict.get(node['type'], len(node_types) - 1) # Use last index for unknown types
        x_type[node['id'], node_type_idx] = 1


    embedding_size = embedding_model.config.hidden_size
    x_name_embedding = torch.zeros(len(nodes), embedding_size)

    for node in nodes:
        if 'function_name' in node:
            try:
                inputs = tokenizer(node['function_name'], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    name_embedding = embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze()
                    x_name_embedding[node['id']] = name_embedding
            except Exception as e:
                print(f"Error generating embedding for function name '{node['function_name']}': {e}")



    x = torch.cat((x_type, x_name_embedding), dim=1)


    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))

# Check for recursion
def is_recursive(code, func_name):
    pattern = rf'\b{func_name}\b'
    match = re.search(pattern, code, re.MULTILINE)
    if match:
        occurrences = [m.start() for m in re.finditer(pattern, code)]
        for occ in occurrences:

            if occ + len(func_name) < len(code) and not code[occ + len(func_name):].lstrip().startswith('def'):
                 return 1
    return 0

# Construct dataset from CodeSearchNet
def load_codesearchnet_dataset(max_samples=1000):
    dataset = load_dataset("Nan-Do/code-search-net-python", split=f"train[:{max_samples*2}]")
    graphs = []
    processed_samples = 0

    for i, sample in enumerate(dataset):
        if processed_samples >= max_samples:
            break

        code = sample.get("code")
        func_name = sample.get("func_name")



        if not code or not func_name:
            continue

        # Parse to AST
        try:
            tree = parser.parse(code.encode())
            if not tree.root_node:
                 continue

            nodes, edges = extract_ast_nodes(tree.root_node, code_text=code)


            if not nodes:
                 continue


            label = is_recursive(code, func_name)


            graph = create_graph(nodes, edges, label)
            graphs.append(graph)
            processed_samples += 1

        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue

    return graphs

print("Loading dataset with enhanced features...")
dataset = load_codesearchnet_dataset(max_samples=1000)
print(f"Dataset size: {len(dataset)} graphs")
if len(dataset) > 0:
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("DataLoader created successfully.")
else:
    print("No data loaded, DataLoader not created.")




class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout_prob=0.5):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_prob)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout_prob)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout_prob)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        return x


in_channels = len(node_types) + embedding_model.config.hidden_size
gat_model = GAT(in_channels=in_channels, hidden_channels=64, out_channels=2, heads=4, dropout_prob=0.5)
gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=0.001)
gat_criterion = torch.nn.CrossEntropyLoss()

def train_gat():
    gat_model.train()
    total_loss = 0
    for data in loader:
        gat_optimizer.zero_grad()
        out = gat_model(data)
        loss = gat_criterion(out, data.y)
        loss.backward()
        gat_optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

print("Starting training of the modified GAT model")
gat_loss_history = []
for epoch in range(50):
    loss = train_gat()
    gat_loss_history.append(loss)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

print("Training complete.")

def infer_recursion(code_snippet, model, parser, tokenizer, embedding_model, node_type_dict):

    try:
        tree = parser.parse(code_snippet.encode())
        if not tree.root_node:
            print("Error: Could not parse code snippet.")
            return -1

        nodes, edges = extract_ast_nodes(tree.root_node, code_text=code_snippet)

        if not nodes:
            print("Error: No nodes extracted from code snippet.")
            return -1

        graph = create_graph(nodes, edges, label=-1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        graph = graph.to(device)

        model.eval()


        with torch.no_grad():
            output = model(graph)
            _, predicted_class = torch.max(output, dim=1)

        return predicted_class.item()

    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return -1