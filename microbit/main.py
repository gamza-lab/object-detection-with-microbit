# USB 시리얼을 열고 명령을 받기 전의 기본 표정을 표시합니다.
serial.redirect_to_usb()
serial.set_baud_rate(BaudRate.BAUD_RATE115200)
basic.show_icon(IconNames.ASLEEP)
last_received = input.running_time()


# PC 앱이 보낸 탐지 여부에 따라 LED 표정을 갱신합니다.
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


# 통신이 끊겨 마지막 표정이 남지 않도록 2초 뒤 기본 상태로 복귀합니다.
def on_forever():
    if input.running_time() - last_received > 2000:
        basic.show_icon(IconNames.ASLEEP, 1)
    basic.pause(100)


basic.forever(on_forever)
