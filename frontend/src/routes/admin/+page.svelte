<script lang="ts">
	const API_URL = 'http://127.0.0.1:8000';

	let loading = $state(true);
	let error = $state('');

	let metrics = $state<any>(null);
	let predictions = $state<any[]>([]);
	let evaluations = $state<any[]>([]);
	let models = $state<any[]>([]);

	async function loadDashboard() {
		loading = true;
		error = '';

		try {
			const [
				metricsResponse,
				predictionsResponse,
				evaluationsResponse,
				modelsResponse
			] = await Promise.all([
				fetch(`${API_URL}/admin/metrics`),
				fetch(`${API_URL}/admin/predictions?limit=20`),
				fetch(`${API_URL}/admin/evaluations?limit=20`),
				fetch(`${API_URL}/admin/models`)
			]);

			if (
				!metricsResponse.ok ||
				!predictionsResponse.ok ||
				!evaluationsResponse.ok ||
				!modelsResponse.ok
			) {
				throw new Error('Failed to load admin data.');
			}

			metrics = await metricsResponse.json();

			const predictionsData =
				await predictionsResponse.json();

			predictions = predictionsData.predictions ?? [];

			const evaluationsData =
				await evaluationsResponse.json();

			evaluations = evaluationsData.evaluations ?? [];

			const modelsData =
				await modelsResponse.json();

			models = modelsData.models ?? [];
		} catch (err) {
			error =
				err instanceof Error
					? err.message
					: 'Failed to load dashboard.';
		} finally {
			loading = false;
		}
	}

	loadDashboard();
</script>

<svelte:head>
	<title>Admin Dashboard · Energy Intelligence</title>
</svelte:head>

<div class="min-h-screen bg-slate-950 text-white">
	<header class="border-b border-white/10">
		<div
			class="mx-auto flex max-w-7xl items-center justify-between px-6 py-5"
		>
			<div>
				<div class="font-semibold tracking-tight">
					Energy Intelligence
				</div>

				<div class="text-xs text-slate-400">
					Production Monitoring
				</div>
			</div>

			<div class="flex items-center gap-3">
				<a
					href="/"
					class="rounded-lg border border-white/10 px-4 py-2 text-sm text-slate-300 transition hover:bg-white/5"
				>
					← Application
				</a>

				<button
					onclick={loadDashboard}
					class="rounded-lg bg-emerald-400 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300"
				>
					Refresh
				</button>
			</div>
		</div>
	</header>

	<main class="mx-auto max-w-7xl px-6 py-10">
		<div class="mb-10">
			<div
				class="mb-3 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/5 px-3 py-1 text-xs font-medium text-emerald-300"
			>
				<span class="h-2 w-2 rounded-full bg-emerald-400"></span>
				PRODUCTION MONITORING
			</div>

			<h1 class="text-4xl font-bold tracking-tight">
				Admin Dashboard
			</h1>

			<p class="mt-3 max-w-2xl text-slate-400">
				Monitor predictions, evaluate production models and
				inspect model performance.
			</p>
		</div>

		{#if error}
			<div
				class="mb-6 rounded-xl border border-red-400/20 bg-red-400/5 p-4 text-sm text-red-300"
			>
				{error}
			</div>
		{/if}

		{#if loading}
			<div
				class="rounded-2xl border border-white/10 bg-white/[0.04] p-10 text-center text-slate-400"
			>
				Loading monitoring data...
			</div>
		{:else if metrics}
			<section class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="text-xs tracking-widest text-slate-500">
						TOTAL PREDICTIONS
					</div>

					<div class="mt-3 text-4xl font-bold">
						{metrics.predictions.total}
					</div>
				</div>

				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="text-xs tracking-widest text-slate-500">
						REGRESSION
					</div>

					<div class="mt-3 text-4xl font-bold">
						{metrics.predictions.regression}
					</div>

					<div class="mt-2 text-sm text-slate-500">
						{metrics.regression.evaluated_predictions}
						evaluated
					</div>
				</div>

				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="text-xs tracking-widest text-slate-500">
						CLASSIFICATION
					</div>

					<div class="mt-3 text-4xl font-bold">
						{metrics.predictions.classification}
					</div>

					<div class="mt-2 text-sm text-slate-500">
						{metrics.classification.evaluated_predictions}
						evaluated
					</div>
				</div>

				<div
					class="rounded-2xl border border-emerald-400/20 bg-emerald-400/5 p-6"
				>
					<div class="text-xs tracking-widest text-emerald-300">
						SYSTEM STATUS
					</div>

					<div class="mt-3 text-2xl font-bold text-emerald-400">
						HEALTHY
					</div>

					<div class="mt-2 text-sm text-slate-500">
						Models operating in production
					</div>
				</div>
			</section>

			<section class="mt-6 grid gap-4 lg:grid-cols-2">
				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="text-xs tracking-widest text-emerald-300">
						REGRESSION EVALUATION
					</div>

					<h2 class="mt-2 text-2xl font-semibold">
						LSTM Production Performance
					</h2>

					<div class="mt-6 grid grid-cols-2 gap-4">
						<div class="rounded-xl bg-black/20 p-5">
							<div class="text-xs text-slate-500">
								MAE
							</div>

							<div class="mt-2 text-3xl font-bold">
								{metrics.regression.mae !== null
									? metrics.regression.mae.toFixed(2)
									: '—'}
							</div>

							<div class="mt-1 text-xs text-slate-500">
								EUR/MWh
							</div>
						</div>

						<div class="rounded-xl bg-black/20 p-5">
							<div class="text-xs text-slate-500">
								RMSE
							</div>

							<div class="mt-2 text-3xl font-bold">
								{metrics.regression.rmse !== null
									? metrics.regression.rmse.toFixed(2)
									: '—'}
							</div>

							<div class="mt-1 text-xs text-slate-500">
								EUR/MWh
							</div>
						</div>
					</div>
				</div>

				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="text-xs tracking-widest text-emerald-300">
						CLASSIFICATION EVALUATION
					</div>

					<h2 class="mt-2 text-2xl font-semibold">
						Dense Neural Network
					</h2>

					<div class="mt-6 rounded-xl bg-black/20 p-5">
						<div class="text-xs text-slate-500">
							ACCURACY
						</div>

						<div class="mt-2 text-4xl font-bold">
							{metrics.classification.accuracy !== null
								? (
										metrics.classification.accuracy * 100
									).toFixed(2) + '%'
								: '—'}
						</div>

						<div class="mt-2 text-sm text-slate-500">
							Based on evaluated production predictions
						</div>
					</div>
				</div>
			</section>

			<section class="mt-6">
				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="mb-5">
						<div class="text-xs tracking-widest text-emerald-300">
							MODEL REGISTRY
						</div>

						<h2 class="mt-2 text-2xl font-semibold">
							Production Models
						</h2>
					</div>

					<div class="grid gap-4 md:grid-cols-2">
						{#each models as model}
							<div
								class="rounded-xl border border-white/10 bg-black/20 p-5"
							>
								<div
									class="flex items-center justify-between"
								>
									<div>
										<div class="font-semibold">
											{model.name}
										</div>

										<div class="mt-1 text-sm text-slate-500">
											{model.type}
										</div>
									</div>

									<span
										class="rounded-full border border-emerald-400/20 bg-emerald-400/5 px-3 py-1 text-xs text-emerald-300"
									>
										{model.status}
									</span>
								</div>

								<div class="mt-4 text-sm text-slate-400">
									Task:
									<span class="text-white">
										{model.task}
									</span>
								</div>

								<div class="mt-1 text-sm text-slate-400">
									Zones:
									<span class="text-white">
										{model.zones.join(', ')}
									</span>
								</div>
							</div>
						{/each}
					</div>
				</div>
			</section>

			<section class="mt-6">
				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="mb-5">
						<div class="text-xs tracking-widest text-emerald-300">
							RECENT PREDICTIONS
						</div>

						<h2 class="mt-2 text-2xl font-semibold">
							Production Activity
						</h2>
					</div>

					{#if predictions.length === 0}
						<div class="py-10 text-center text-slate-500">
							No predictions recorded yet.
						</div>
					{:else}
						<div class="overflow-x-auto">
							<table class="w-full text-left text-sm">
								<thead>
									<tr
										class="border-b border-white/10 text-xs uppercase tracking-wider text-slate-500"
									>
										<th class="px-4 py-3">Zone</th>
										<th class="px-4 py-3">Model</th>
										<th class="px-4 py-3">Prediction</th>
										<th class="px-4 py-3">Time</th>
									</tr>
								</thead>

								<tbody>
									{#each predictions as item}
										<tr
											class="border-b border-white/5"
										>
											<td class="px-4 py-4">
												{item.zone}
											</td>

											<td class="px-4 py-4 text-slate-400">
												{item.model_type}
											</td>

											<td class="px-4 py-4 font-semibold">
												{item.prediction ??
													item.prediction_class}
											</td>

											<td class="px-4 py-4 text-slate-500">
												{item.created_at}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{/if}
				</div>
			</section>

			<section class="mt-6">
				<div
					class="rounded-2xl border border-white/10 bg-white/[0.04] p-6"
				>
					<div class="mb-5">
						<div class="text-xs tracking-widest text-emerald-300">
							EVALUATIONS
						</div>

						<h2 class="mt-2 text-2xl font-semibold">
							Production Model Evaluation
						</h2>
					</div>

					{#if evaluations.length === 0}
						<div class="py-10 text-center text-slate-500">
							No production evaluations yet.
						</div>
					{:else}
						<div class="overflow-x-auto">
							<table class="w-full text-left text-sm">
								<thead>
									<tr
										class="border-b border-white/10 text-xs uppercase tracking-wider text-slate-500"
									>
										<th class="px-4 py-3">Model</th>
										<th class="px-4 py-3">Zone</th>
										<th class="px-4 py-3">Predicted</th>
										<th class="px-4 py-3">Actual</th>
										<th class="px-4 py-3">Result</th>
									</tr>
								</thead>

								<tbody>
									{#each evaluations as evaluation}
										<tr
											class="border-b border-white/5"
										>
											<td class="px-4 py-4">
												{evaluation.model_type}
											</td>

											<td class="px-4 py-4">
												{evaluation.zone}
											</td>

											<td class="px-4 py-4">
												{evaluation.predicted_value ??
													evaluation.predicted_class}
											</td>

											<td class="px-4 py-4">
												{evaluation.actual_value ??
													evaluation.actual_class}
											</td>

											<td class="px-4 py-4">
												{#if evaluation.model_type === 'regression'}
													<span class="text-slate-400">
														Error:
														{evaluation.absolute_error?.toFixed(
															2
														)}
													</span>
												{:else}
													{#if evaluation.is_correct}
														<span class="text-emerald-400">
															Correct
														</span>
													{:else}
														<span class="text-red-400">
															Incorrect
														</span>
													{/if}
												{/if}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{/if}
				</div>
			</section>
		{/if}
	</main>
</div>