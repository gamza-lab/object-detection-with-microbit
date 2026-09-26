# MakeCode의 Python 편집기에 붙여 넣으세요.
serial.redirect_to_usb()
serial.set_baud_rate(BaudRate.BAUD_RATE115200)
basic.show_icon(IconNames.ASLEEP)
last_received = input.running_time()


def on_data_received():
    global last_received
    command = serial.read_until(serial.delimiters(Delimiters.NEW_LINE))
    if command == "1" or command == "0":
        last_received = input.running_time()
        if command == "1":
            basic.show_icon(IconNames.HAPPY, 1)
        else:
            basic.show_icon(IconNames.ASLEEP, 1)


serial.on_data_received(serial.delimiters(Delimiters.NEW_LINE), on_data_received)


def on_forever():
    if input.running_time() - last_received > 2000:
        basic.show_icon(IconNames.ASLEEP, 1)
    basic.pause(100)


basic.forever(on_forever)
