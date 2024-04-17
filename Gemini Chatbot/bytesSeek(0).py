import io

data = b'Hello, world!'

# Create a file-like object from the bytes object
file_like_object = io.BytesIO(data)

# Move the file pointer to the beginning of the file
file_like_object.seek(0)

# Read the contents of the file
contents = file_like_object.read()

print(contents)