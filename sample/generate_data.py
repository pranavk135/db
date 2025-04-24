import random

def generate_data(num_records, filename):
    with open(filename, 'w') as file:
        file.write("feature1\tfeature2\tfeature3\tlabel\n")
        for _ in range(num_records):
            feature1 = round(random.uniform(0, 10), 1)
            feature2 = round(random.uniform(0, 10), 1)
            feature3 = round(random.uniform(0, 10), 1)
            label = random.choice(['normal', 'botnet'])
            file.write(f"{feature1}\t{feature2}\t{feature3}\t{label}\n")

if __name__ == "__main__":
    generate_data(100, 'network_traffic_data.txt')
