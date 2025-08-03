package SumOfNumbers;

import java.util.Scanner;

public class Sum {
    public static void main(String[] args) {
        // Create a Scanner object to read input
        Scanner scanner = new Scanner(System.in);

        // Ask user for comma-separated numbers
        System.out.print("Enter comma-separated numbers: ");
        String input = scanner.nextLine();

        if (input.trim().isEmpty() || input.trim().equals(",")) {
            System.out.println("Invalid input.");
            scanner.close();
            return;
        }

        // Split the input string into an array of strings based on commas
        String[] numbers = input.split(",");

        // Initialize a variable to store the sum
        int sum = 0;

        // Loop through each number in the array
        for (String number : numbers) {
            // Remove any leading/trailing whitespace and parse each number
            (if number != null && !number.trim().isEmpty()) {
                continue;
            }
            sum += Integer.parseInt(number.trim());
        }

        // Output the sum
        System.out.println("The sum is: " + sum);

        // Close the scanner to prevent memory leaks
        scanner.close();
    }
}