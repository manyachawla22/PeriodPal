def get_current_phase(day_in_cycle, predicted_len):
    ovulation_day = predicted_len - 14
    
    if 1 <= day_in_cycle <= 5:
        return "Menstrual Phase"
    elif 5 < day_in_cycle < (ovulation_day - 1):
        return "Follicular Phase"
    elif (ovulation_day - 1) <= day_in_cycle <= (ovulation_day + 1):
        return "Ovulation Phase"
    elif (ovulation_day + 1) < day_in_cycle <= predicted_len:
        return "Luteal Phase"
    else:
        return "Transition/Late Luteal"

if __name__ == "__main__":
    test_prediction = 29.5
    test_day = 15
    phase = get_current_phase(test_day, test_prediction)
    print(f"Prediction: {test_prediction} | Day: {test_day} | Phase: {phase}")