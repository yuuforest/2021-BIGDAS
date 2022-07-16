package Problem.D2;

import java.util.Scanner;

public class Prob1959 {
    public static int calculate(int[] S, int[] B) {

		int sum = 0;
		int temp = 0;

		int s = S.length;
		int b = B.length;

		for (int i = 0; i < (b - s + 1); i++) {
			temp = 0;
			for (int j = 0; j < s; j++) temp += S[j]*B[i+j];
			sum = Math.max(sum, temp);
		}

		return sum;
	}

	public static void main(String args[]) throws Exception{
		
		Scanner sc = new Scanner(System.in);
		
		int T = sc.nextInt();

		for (int testcase = 1; testcase < T+1; testcase++) {

			int N = sc.nextInt();
			int M = sc.nextInt();

			int[] A = new int[N];
			int[] B = new int[M];

			for (int j = 0; j < N; j++) A[j] = sc.nextInt();
			for (int j = 0; j < M; j++) B[j] = sc.nextInt();

			int max = 0;

			if (N == M) for (int i = 0; i < N; i++) max += A[i]*B[i]; 
			else if (N < M) max = calculate(A, B);
			else max = calculate(B, A);

			System.out.println("#" + testcase + " " + max);
		}
		sc.close();
	}
}
