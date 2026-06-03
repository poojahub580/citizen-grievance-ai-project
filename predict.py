complaint = input("Enter complaint: ")

if "repaired" in complaint.lower():
    print("Predicted Sentiment: Positive")
else:
    print("Predicted Sentiment: Critical")
    