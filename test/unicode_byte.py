def bytes_to_unicode():
    """
    生成字节到Unicode字符的正向映射表
    返回字典：{byte_value: unicode_char}
    """
    # 原始保留的字节范围
    bs = (
        list(range(ord("!"), ord("~") + 1)) +          # ASCII可打印字符（33-126）
        list(range(ord("¡"), ord("¬") + 1)) +          # 西班牙语特殊字符（161-172）
        list(range(ord("®"), ord("ÿ") + 1))            # 其他扩展字符（174-255）
    )
    
    cs = bs.copy()  # 初始字符列表
    n = 0
    
    # 遍历所有可能的字节（0-255）
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)  # 超出原始范围的字节映射到更高Unicode码位
            n += 1
    
    # 将码位转换为Unicode字符
    cs = [chr(code) for code in cs]
    
    return dict(zip(bs, cs))

def get_reverse_mapping(forward_map):
    """
    根据正向映射生成反向映射
    返回字典：{unicode_char: byte_value}
    """
    return {v: k for k, v in forward_map.items()}

# 生成映射表
forward_map = bytes_to_unicode()
reverse_map = get_reverse_mapping(forward_map)

unicode_map = {ord(char): byte for byte, char in forward_map.items()}

print(unicode_map[322])
print(forward_map[unicode_map[322]])

# 查看字节值136的映射
# byte_val = 136
# print(f"字节 {byte_val} 对应的Unicode字符是：{forward_map[byte_val]}") 
# 输出：字节 136 对应的Unicode字符是：Ī

def bytes_to_unicode_str(byte_sequence):
    return ''.join([forward_map[b] for b in byte_sequence])

# 将"你好"的UTF-8字节转换为Unicode字符串
# text = "你好"
# byte_sequence = text.encode('utf-8')
# print()
# unicode_str = bytes_to_unicode_str(byte_sequence)
# print(f"原始文本：{text}")
# print(f"转换后的Unicode字符串：{unicode_str}")
# 输出：
# 原始文本：你好
# 转换后的Unicode字符串：ä½łå¥½
def unicode_str_to_bytes(unicode_str):
    return bytes([reverse_map[c] for c in unicode_str])


unicode_str = 'ä½łå¥½'
# 将转换后的Unicode字符串还原为原始文本


for char in unicode_str:
    code_point = ord(char)
    bin_without_prefix = format(code_point, 'b')  # 无前缀的二进制
    print(f"字符: {char}, 十进制码点: {code_point}, uft-8值: {'/'.join([format(byte, '08b') for byte in bytes(char.encode('utf-8'))])}, 无前缀二进制: {bin_without_prefix}, 映射后的字节: {unicode_map[code_point]}")


recovered_bytes = unicode_str_to_bytes(unicode_str)


decimal_list = list(recovered_bytes)
print(decimal_list)

recovered_text = recovered_bytes.decode('utf-8')
# print(f"还原后的文本：{recovered_text}")
# for char in recovered_text:
#     code_point = ord(char)
#     bin_without_prefix = format(code_point, 'b')  # 无前缀的二进制
#     print(f"字符: {char}, 十进制码点: {code_point}, uft-8值: {'/'.join([format(byte, '08b') for byte in bytes(char.encode('utf-8'))])}, 无前缀二进制: {bin_without_prefix}")

# 输出：还原后的文本：你好

