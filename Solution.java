
import java.io.FileInputStream;
import java.util.Scanner;

class Solution{

	public static void main(String args[]) throws Exception{
		
		System.setIn(new FileInputStream("C:/Users/Yu/Desktop/VSCode/JAVA/SWExpertAcademy/src/input.txt"));
		Scanner sc = new Scanner(System.in);
		
		int T = sc.nextInt();

		for (int testcase = 1; testcase < T+1; testcase++) {

			

			System.out.println("#" + testcase);
		}
		sc.close();
	}
}
