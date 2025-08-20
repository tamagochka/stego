


# emb = []
# with open('embeding.txt', 'r') as F:
#     emb = F.readlines()
# print(f'emb len: {len(emb)}')

# ext = []
# with open('extracting.txt', 'r') as F:
#     ext = F.readlines()
# print(f'ext len: {len(ext)}')

# eq = 0
# for i in range(len(emb)):
#     if ('>' in emb[i] and '>' in ext[i]) or \
#         ('<' in emb[i] and '<' in ext[i]) or \
#         ('-' in emb[i] and '-' in ext[i]):
#         eq += 1
#     else:
#         print(i)

# print(f'eq: {eq}')


from numpy import zeros, uint8


msg = [0, 0, 0, 1, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 0, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 0, 1, 0, 0, 1, 0,
       0, 0, 0, 0, 0, 0, 1, 0,
       0, 0, 1, 0, 1, 1, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 0]

tmp = zeros(8, dtype=uint8)
for i in range(len(msg)):
    # tmp[(i + 1) % 8 - 1] = msg[i]
    # print(f'{(i - 1) % 8 + 1} - {tmp[(i + 1) % 8 - 1]}')
    # if not (i + 1) % 8:
    #     print(tmp)

    tmp[i % 8] = msg[i]
    if not (i + 1) % 8:
        val = 0
        for b in tmp:
            val = (val << 1) | b
        print(val)


    # print(f'{i % 8}')
    # if (i + 1) % 8 == 0:
    #     print('---')


