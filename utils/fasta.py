def read_fasta(fasta_file):
    fasta_dict = {}
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                seq_name = line.strip()[1:]
                fasta_dict[seq_name] = ""
            else:
                fasta_dict[seq_name] += line.strip()
    fasta_list = [(seq_name, seq) for seq_name, seq in fasta_dict.items()]
    return fasta_list
