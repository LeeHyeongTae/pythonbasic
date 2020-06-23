from calculator.controller import Controller

if __name__ == '__main__':
    def print_menu():
        print('0. Exit')
        print('1. Calculator')
        return input('Menu\n')

    while 1:
        menu = print_menu()
        if menu == '0': break
        if menu == '1':
            app = Controller()
            print('계산기 작동')
            num1 = int(input('첫번째 수\n'))
            opcode = str(input('연산\n'))
            num2 = int(input('두번째 수\n'))

            result = app.calc(num1, num2, opcode)
            print('결과: %d' % result)
        if menu == '2': break
