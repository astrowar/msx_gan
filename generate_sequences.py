import torch

# Parâmetros
num_sequences = 8192
sequence_length = 64
multiplier = 16

# Gerar sequências
sequences = torch.randn(num_sequences, sequence_length) * multiplier

# Salvar em arquivo c como um array de inteiros
with open('random_sequences.c', 'w') as f:
    f.write('int sequences[{}][{}] = {{\n'.format(num_sequences, sequence_length))
    for seq in sequences:
        f.write('    {' + ', '.join(f'{int(v)}' for v in seq.tolist()) + '},\n')
    f.write('};\n')

print(f'Sequências salvas em random_sequences.c')
