import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/route_request_model.dart';
import '../models/route_response_model.dart';
import '../services/route_api_service.dart';

class RouteNotifier extends StateNotifier<AsyncValue<RouteResponse?>> {
  RouteNotifier() : super(const AsyncValue.data(null));

  Future<void> getRoute({
    required double startLat,
    required double startLng,
    required double endLat,
    required double endLng,
  }) async {
    state = const AsyncValue.loading();

    try {
      final request = RouteRequest(
        startLat: startLat,
        startLng: startLng,
        endLat: endLat,
        endLng: endLng,
      );

      final response = await RouteApiService.getRoute(request);
      state = AsyncValue.data(response);
    } catch (error, stackTrace) {
      state = AsyncValue.error(error, stackTrace);
    }
  }

  void clearRoute() {
    state = const AsyncValue.data(null);
  }
}

final routeProvider =
    StateNotifierProvider<RouteNotifier, AsyncValue<RouteResponse?>>((ref) {
      return RouteNotifier();
    });
