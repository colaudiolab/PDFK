# Create a colored bar chart per user's data and save to /mnt/data
import matplotlib.pyplot as plt

labels = [
    "ER",  "ER + SDP",
    "DER++",
    "ERACE",
    "DVC",
    "OCM",
    "GSA",  "GSA + SDP", "Ours"
]

times = [10, 19, 18, 22, 17, 46, 38, 45, 26]

# Use distinct colors from matplotlib's tab10 palette
colors = [plt.cm.tab10(i % 10) for i in range(len(labels))]

plt.figure(figsize=(8, 4))
plt.bar(labels, times, color=colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Time Consumption (minutes)")
plt.tight_layout()

pdf_path = "/mnt/data/time_consumption_cifar10_m500_4090D.pdf"
# png_path = "/mnt/data/time_consumption_cifar10_m500_4090D.png"
plt.savefig(pdf_path, dpi=300)
# plt.savefig(png_path, dpi=300)
plt.close()

pdf_path
