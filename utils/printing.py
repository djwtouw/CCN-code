def pretty_label(label):
    # Capitalize first letter
    temp = label[0].capitalize() + label[1:]

    # Replace underscores by dashes if they are followed by a lower case letter
    # and a space if they are followed by an upper case letter
    result = ""
    for i in range(len(temp)):
        if temp[i] == "_" and i < len(temp) - 1:
            result += " " if temp[i + 1].isupper() else "-"
        else:
            result += temp[i]

    return result
