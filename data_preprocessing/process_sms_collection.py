def process_data(file_path):
    file = open(file_path)
    lines = file.readlines()
    file.close()
    processed_lines = []
    for line in lines:
        line = line.replace(',', '')
        if line.startswith('ham'):
            index = line.find('ham')
            position = index + len('ham')
            line = line[0: position] + '::' + line[position:].strip() + '\n'
            processed_lines.append(line)
        elif line.startswith('spam'):
            index = line.find('spam')
            position = index + len('spam')
            line = line[0: position] + '::' + line[position:].strip() + '\n'
            processed_lines.append(line)
    file = open(file_path, 'w')
    file.writelines(processed_lines)


if __name__ == '__main__':
    process_data('/datasets/SMSSpamCollection.csv')
