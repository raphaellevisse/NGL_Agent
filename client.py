import socket

def test_connection(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_socket.connect((host, port))
        print(f"Connected to server at {host}:{port}")

        client_socket.sendall(b"Hello, from external computer!")
        data = client_socket.recv(1024)
        print(f"Received from server: {data.decode()}")
    
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    host = "10.48.33.107"  # Replace with the server's IP address
    port = 12345  # Make sure it matches the server's port
    test_connection(host, port)
