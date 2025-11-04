
emb_arr: list[str] = []
with open('emb_edge.txt', 'r') as f:
    for line in f:
        emb_arr.append(line)

ext_arr: list[str] = []
with open('ext_edge.txt', 'r') as f:
    for line in f:
        ext_arr.append(line)

with open('diff.txt', 'w') as f:
    for i in range(len(emb_arr)):
        if ('emb' in emb_arr[i]) or ('emb' in ext_arr[i]):
            f.write(f'{emb_arr[i].strip()} - {ext_arr[i].strip()}')
            
            if not('emb' in emb_arr[i] and 'emb' in ext_arr[i]):
                f.write(f' - diff')

            f.write('\n')



