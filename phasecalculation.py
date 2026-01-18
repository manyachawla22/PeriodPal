def get_current_phase(day_in_cycle, total_cycle_length):
    
    ovulation_day = total_cycle_length - 14
    
    if 1 <= day_in_cycle <= 5:
        return "Menstrual Phase"
    elif 5 < day_in_cycle < (ovulation_day - 1):
        return "Follicular Phase"
    elif (ovulation_day - 1) <= day_in_cycle <= (ovulation_day + 1):
        return "Ovulation Phase"
    else:
        return "Luteal Phase"

if __name__ == "__main__":
    predicted_len = 30.7
    test_day = 14
    
    phase = get_current_phase(test_day, predicted_len)
    print(f"On Day {test_day} of a {round(predicted_len)} day cycle, you are in the: {phase}")